import argparse
import itertools
import math

import numpy as np
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    bbox_diagonal,
    cast_ray_bundle_with_clearance,
    prompt_for_clearance_rings,
    rotate_about_axis,
)
from weld_line_finder import (
    identify_components,
    collect_candidate_edges_assembly,
    resolve_assembly_chain_faces,
    find_weld_lines_interactive,
    visualize_weld_line_accessibility,
)

# Full brute-force enumeration (every ordering gets its own full report) is used up to this many
# weld lines; beyond it the factorial blow-up makes printing (and even just iterating) every
# ordering impractical, so only the guaranteed-optimal sequence is found, via a subset DP.
MAX_FULL_ENUMERATION = 8

# TEMPORARY MANUAL OVERRIDE -- set back to 0.0 to fully revert to testing the pure bisector.
# Nonzero: every accessibility check in the search tests a ray tilted this many degrees up/down
# (work angle, rotated about each point's own tangent -- same convention as weld_angle_updown.py)
# instead of the ideal bisector. Positive/negative sign picks which side it tilts toward.
MANUAL_TEST_TILT_DEG = 0


def _test_directions(wl):
    """The ray direction(s) actually tested for a weld line -- the bisector, unless
    MANUAL_TEST_TILT_DEG overrides it. Used by both the search (SequenceEvaluator.cost) and the
    optional final visualization, so the two can never silently disagree with each other."""
    if MANUAL_TEST_TILT_DEG == 0:
        return wl["bisectors"]
    return rotate_about_axis(wl["bisectors"], wl["tangents"], MANUAL_TEST_TILT_DEG)


def prompt_num_samples(default=15):
    raw = input(f"Number of sample points to test per weld line [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        print(f"Couldn't parse that -- using default ({default}).")
        return default


def weld_own_other_components(weld_lines, component_of_face):
    """Each weld line's own/other component is constant across all its pieces by construction
    (resolve_assembly_chain_faces asks for the other component exactly once per weld line and
    enforces a single own component), so the first piece's faces are enough to read it off."""
    pairs = []
    for wl in weld_lines:
        own_face, other_face = wl["per_piece_faces"][0][0][0], wl["per_piece_faces"][0][1][0]
        pairs.append((component_of_face[own_face], component_of_face[other_face]))
    return pairs


def weld_label(idx, own_comp, other_comp):
    return f"W{idx}(comp{own_comp}<->comp{other_comp})"


class SequenceEvaluator:
    """Ray-casts each weld line only against the components physically present at that point in
    a candidate sequence, memoized by (present-component-set, weld) so the same physical
    situation is never ray-cast twice even when it recurs across many different orderings.

    A point only counts as accessible if the bisector ray AND every clearance-ring ray around it
    are clear (see cast_ray_bundle_with_clearance) -- the same spherical-tool simulation already
    used by the other scripts' visualizations -- so the search itself accounts for the torch
    having a real radius, not just an infinitely thin ray, instead of only checking that
    afterward in the optional final visualization."""

    def __init__(self, tagged_mesh, weld_lines, own_other_components, baseline_present, near_tol,
                 n_rings=0, clearance_radius=5.0, ring_offset=None):
        self.tagged_mesh = tagged_mesh
        self.weld_lines = weld_lines
        self.own_other = own_other_components
        self.baseline_present = frozenset(baseline_present)
        self.near_tol = near_tol
        self.n_rings = n_rings
        self.clearance_radius = clearance_radius
        self.ring_offset = ring_offset
        self.max_dist = bbox_diagonal(tagged_mesh)
        self._mesh_cache = {}
        self._cost_cache = {}

    def obstruction_mesh(self, present_components):
        key = frozenset(present_components)
        mesh = self._mesh_cache.get(key)
        if mesh is None:
            mask = np.isin(self.tagged_mesh.cell_data["ComponentID"], list(key))
            mesh = self.tagged_mesh.extract_cells(mask).extract_surface()
            self._mesh_cache[key] = mesh
        return mesh

    def cost(self, weld_idx, present_components):
        key = (frozenset(present_components), weld_idx)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        wl = self.weld_lines[weld_idx]
        mesh = self.obstruction_mesh(present_components)
        directions = _test_directions(wl)
        combined_accessible, _center, _center_hits, _ring_acc, _ring_hits, _ring_o, _ring_d = \
            cast_ray_bundle_with_clearance(
                mesh, wl["points"], directions, self.max_dist, self.near_tol,
                self.n_rings, self.clearance_radius, ring_offset=self.ring_offset)
        n_blocked = int((~combined_accessible).sum())
        result = (n_blocked, len(wl["points"]))
        self._cost_cache[key] = result
        return result

    def evaluate_sequence(self, order):
        present = set(self.baseline_present)
        steps = []
        total_blocked = 0
        for widx in order:
            own, other = self.own_other[widx]
            present.add(own); present.add(other)
            n_blocked, n_total = self.cost(widx, present)
            total_blocked += n_blocked
            steps.append({
                "weld": widx, "present": sorted(present),
                "n_blocked": n_blocked, "n_total": n_total, "is_full": n_blocked == 0,
            })

        # A joint between two bodies that had to be entered as several disconnected weld-line
        # pieces (non-continuous edges can't be one pick) is judged as ONE unit here: it only
        # counts toward fully_count if EVERY one of its pieces -- wherever each lands in the
        # sequence -- ends up with zero blocked rays. Otherwise a joint split into many small,
        # individually-clear pieces would inflate this count against an equally-good joint that
        # just happened to be pickable as a single piece.
        blocked_by_pair = {}
        for step in steps:
            key = frozenset(self.own_other[step["weld"]])
            blocked_by_pair[key] = blocked_by_pair.get(key, 0) + step["n_blocked"]
        fully_count = sum(1 for n in blocked_by_pair.values() if n == 0)

        return {"order": list(order), "steps": steps, "fully_count": fully_count, "total_blocked": total_blocked}


def _score(result):
    # Sort key: maximize fully_count (priority 1), then minimize total_blocked (priority 2).
    return (-result["fully_count"], result["total_blocked"])


def enumerate_all_sequences(evaluator, n_welds):
    return [evaluator.evaluate_sequence(order) for order in itertools.permutations(range(n_welds))]


def find_optimal_via_dp(evaluator, n_welds):
    """Subset DP (Held-Karp style): dp[mask] = best (total_blocked, backpointer) for having
    completed exactly the welds in `mask`, in the best order found so far. Used instead of brute
    force once n_welds exceeds MAX_FULL_ENUMERATION, since factorial enumeration becomes
    impractical -- this still finds the exact, guaranteed order minimizing total blocked rays, in
    O(2^n * n).

    This optimizes total_blocked only, NOT the grouped fully_count metric (priority 1) that the
    brute-force path uses: fully_count depends on ALL of a body-pair's pieces being 0-blocked,
    wherever each piece lands in the sequence, so it isn't a quantity that can be built up one
    weld at a time the way an incremental DP requires (a partial prefix can't know yet whether a
    pair it has only partly placed will end up fully accessible). It's still reported correctly
    for the resulting order via evaluate_sequence() below -- just not the search objective for
    this fallback path -- so at this scale the two priorities can't both be exactly guaranteed
    optimal simultaneously; total blocked rays is treated as the more scalable proxy."""
    n = n_welds
    present_of_mask = [None] * (1 << n)
    present_of_mask[0] = frozenset(evaluator.baseline_present)
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        prev_mask = mask & ~lsb
        own, other = evaluator.own_other[i]
        present_of_mask[mask] = present_of_mask[prev_mask] | {own, other}

    # dp[mask] = (total_blocked, last_weld, prev_mask)
    dp = [None] * (1 << n)
    dp[0] = (0, None, None)
    for mask in range(1, 1 << n):
        best = None
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            prev_mask = mask & ~(1 << i)
            prev_blocked, _, _ = dp[prev_mask]
            n_blocked, _n_total = evaluator.cost(i, present_of_mask[mask])
            cand_blocked = prev_blocked + n_blocked
            if best is None or cand_blocked < best[0]:
                best = (cand_blocked, i, prev_mask)
        dp[mask] = best

    full_mask = (1 << n) - 1
    order = []
    mask = full_mask
    while mask:
        _, last_weld, prev_mask = dp[mask]
        order.append(last_weld)
        mask = prev_mask
    order.reverse()

    return evaluator.evaluate_sequence(order)


def print_sequence_report(result, evaluator, own_other, title):
    print(f"\n--- {title} ---")
    print(f"Order: {' -> '.join(weld_label(w, *own_other[w]) for w in result['order'])}")
    for step in result["steps"]:
        status = "OK (0 blocked)" if step["is_full"] else f"BLOCKED {step['n_blocked']}/{step['n_total']}"
        print(f"  {weld_label(step['weld'], *own_other[step['weld']])}: present components "
              f"{step['present']} -> {status}")
    print(f"  Summary: {result['fully_count']} body-pair joint(s) fully accessible (all their "
          f"piece(s) combined), {result['total_blocked']} blocked ray(s) total")


def print_ranked_summary(results, own_other, optimal):
    print("\n" + "=" * 78)
    print("ALL SEQUENCES RANKED (best first)")
    print("=" * 78)
    header = f"{'Rank':<5}{'Order':<45}{'PairsOK':<9}{'Blocked':<9}{'vs optimal'}"
    print(header)
    for rank, result in enumerate(sorted(results, key=_score), start=1):
        order_str = ",".join(f"W{w}" for w in result["order"])
        d_fully = result["fully_count"] - optimal["fully_count"]
        d_blocked = result["total_blocked"] - optimal["total_blocked"]
        delta = "OPTIMAL" if result is optimal or (d_fully == 0 and d_blocked == 0) else \
            f"{d_fully:+d} fully-OK, {d_blocked:+d} blocked"
        print(f"{rank:<5}{order_str:<45}{result['fully_count']:<9}{result['total_blocked']:<9}{delta}")
    print("=" * 78)


def run_sequence_search(tagged_mesh, weld_lines, own_other, baseline_present, near_tol,
                         max_full_enum=MAX_FULL_ENUMERATION, n_rings=0, clearance_radius=5.0,
                         ring_offset=None):
    n = len(weld_lines)
    evaluator = SequenceEvaluator(tagged_mesh, weld_lines, own_other, baseline_present, near_tol,
                                   n_rings=n_rings, clearance_radius=clearance_radius,
                                   ring_offset=ring_offset)
    if n_rings > 0:
        effective_offset = ring_offset if ring_offset is not None else clearance_radius
        print(f"Simulating a spherical tool of radius {clearance_radius:.1f} mm via {n_rings} "
              f"clearance ring ray(s) per point (start offset {effective_offset:.1f} mm forward "
              f"along each ray) for every accessibility check in the search.")

    if n <= max_full_enum:
        print(f"\nEvaluating all {n}! = {math.factorial(n)} possible weld order(s)...")
        results = enumerate_all_sequences(evaluator, n)
        optimal = min(results, key=_score)
        for result in sorted(results, key=_score):
            title = "OPTIMAL SEQUENCE" if result is optimal else "Sequence"
            print_sequence_report(result, evaluator, own_other, title)
        print_ranked_summary(results, own_other, optimal)
    else:
        print(f"\n{n} weld lines -> {n}! orderings is too many to enumerate and print individually "
              f"(limit is {max_full_enum}, set with --max_full_enum). Finding the order that "
              f"minimizes total blocked rays via an exact subset search instead (guaranteed optimal "
              f"for that criterion; the number of fully-accessible body-pair joints is reported "
              f"below for the result but wasn't the search objective at this scale -- see "
              f"find_optimal_via_dp).")
        optimal = find_optimal_via_dp(evaluator, n)
        print_sequence_report(optimal, evaluator, own_other, "OPTIMAL SEQUENCE")
        results = [optimal]

    return optimal, evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Finds weld lines on a STEP assembly (same picking flow as weld_line_finder.py), "
                     "then searches every possible weld ORDER for the one that leaves the fewest weld "
                     "lines obstructed, accounting for the fact that a not-yet-welded part may not be "
                     "physically present yet to block the torch.")
    parser.add_argument("--step", required=True, help="Path to input STEP assembly file")
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Points sampled per weld line (skips the interactive prompt if given)")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Ray start offset (mm)")
    parser.add_argument("--proximity_tol", type=float, default=2.0,
                         help="Max distance (mm) for a face on the chosen other component to be "
                              "accepted as a weld line's far-side face")
    parser.add_argument("--parallel_tol_deg", type=float, default=20.0,
                         help="For a same-component corner edge, an own-side candidate face within "
                              "this many degrees of parallel to the resolved far-side face is treated "
                              "as the hidden flush-contact face and eliminated")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    parser.add_argument("--max_full_enum", type=int, default=MAX_FULL_ENUMERATION,
                         help="Max weld lines to brute-force enumerate every ordering for (exact "
                              "search on both priorities, full per-sequence report). Above this, "
                              "falls back to a subset-DP search that only guarantees minimizing "
                              "total blocked rays (see run_sequence_search's docstring)")
    parser.add_argument("--n_rings", type=int, default=None,
                         help="Clearance ring rays per point simulating a spherical tool radius, "
                              "used for every accessibility check in the search itself (not just "
                              "the optional final visualization). Skips the interactive prompt "
                              "below if given; 0 disables")
    parser.add_argument("--clearance_radius", type=float, default=None,
                         help="Spherical tool radius in mm for the clearance ring rays "
                              "(only used together with --n_rings)")
    parser.add_argument("--ring_offset", type=float, default=None,
                         help="Forward start-point offset (mm) for clearance ring rays "
                              "(default: clearance_radius; only used together with --n_rings)")
    args = parser.parse_args()

    if args.n_rings is not None:
        n_rings = args.n_rings
        clearance_radius = args.clearance_radius if args.clearance_radius is not None else 5.0
        ring_offset = args.ring_offset
    else:
        n_rings, clearance_radius, ring_offset = prompt_for_clearance_rings()

    print("Loading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    solids = list(topo.solids())
    if len(solids) <= 1:
        raise RuntimeError(
            "weld_sequence.py only works on assemblies (2+ separate components) -- this STEP file "
            "has a single solid. Use weld_line_finder.py for a single-body part.")

    solids, faces, component_of_face = identify_components(shape)
    n_components = len(solids)
    print(f"Assembly mode: {n_components} component(s), {len(faces)} face(s) total.")

    face_groups = find_cylinder_face_groups(faces, component_of_face=component_of_face)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)
    lookup = np.array(component_of_face, dtype=np.int32)
    tagged_mesh.cell_data["ComponentID"] = lookup[tagged_mesh.cell_data["FaceID"]]

    candidate_edges, own_info = collect_candidate_edges_assembly(shape, faces, face_groups, component_of_face)
    if not candidate_edges:
        raise RuntimeError("No usable edges were found -- nothing to pick.")

    num_samples = args.num_samples if args.num_samples is not None else prompt_num_samples()

    def resolve_faces_fn(chain):
        return resolve_assembly_chain_faces(
            tagged_mesh, faces, face_groups, component_of_face, n_components,
            chain, candidate_edges, own_info, args.proximity_tol, args.parallel_tol_deg)

    weld_lines = find_weld_lines_interactive(
        tagged_mesh, faces, candidate_edges, num_samples, resolve_faces_fn)

    if not weld_lines:
        print("\nNo weld lines picked -- nothing to sequence.")
        return

    own_other = weld_own_other_components(weld_lines, component_of_face)
    print(f"\n{len(weld_lines)} weld line(s) picked:")
    for i, (own, other) in enumerate(own_other):
        print(f"  {weld_label(i, own, other)}: {len(weld_lines[i]['points'])} sample point(s)")

    if input("\nProceed with the weld-order search on these weld line(s)? [Y/n]: ").strip().lower() in ("n", "no"):
        print("Cancelled.")
        return

    referenced = {c for pair in own_other for c in pair}
    baseline_present = set(range(n_components)) - referenced
    if baseline_present:
        print(f"[Setup] Component(s) {sorted(baseline_present)} aren't the target of any picked weld "
              f"line -- treated as fixed context, present for every step.")

    optimal, evaluator = run_sequence_search(
        tagged_mesh, weld_lines, own_other, baseline_present, args.near_tol, args.max_full_enum,
        n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset)

    if input("\nVisualize the optimal sequence's ray casts, step by step? [y/N]: ").strip().lower() in ("y", "yes"):
        for step in optimal["steps"]:
            wl = weld_lines[step["weld"]]
            print(f"\n--- {weld_label(step['weld'], *own_other[step['weld']])}: "
                  f"present components {step['present']} ---")
            mesh = evaluator.obstruction_mesh(step["present"])
            visualize_weld_line_accessibility(
                mesh, wl["chain"], wl["points"], _test_directions(wl), near_tol=args.near_tol,
                n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset)


if __name__ == '__main__':
    main()
