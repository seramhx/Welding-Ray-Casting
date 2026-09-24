import argparse
import contextlib
import html
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    bbox_diagonal,
    cast_ray_bundle_with_clearance,
    _render_bisector_scene,
    prompt_for_clearance_rings,
)
from weld_line_finder import (
    identify_components,
    collect_candidate_edges_assembly,
    show_labeled_edges,
    build_chains_from_ids,
    sample_and_analyze_chain,
    preview_weld_line,
    _edge_index_in,
    component_face_groups,
    find_closest_face_groups,
    eliminate_parallel_own_face,
)
import weld_sequence
from weld_sequence import (
    weld_own_other_components,
    weld_label,
    prompt_num_samples,
    SequenceEvaluator,
    enumerate_all_sequences,
    find_optimal_via_dp,
    print_ranked_summary,
    _score,
    _test_directions,
)

THIS_DIR = Path(__file__).resolve().parent

# weld_sequence.py's own default (MAX_FULL_ENUMERATION = 8) is sized around also printing a full
# per-order report for every permutation. This script only ever prints the final ranked summary
# table (see run_sequence_search_quiet), so there's room to brute-force one more weld line's worth
# of orderings before falling back to the DP approximation.
DEFAULT_MAX_FULL_ENUM = 10

# Every ray-cast visualization in this pipeline draws its rays out to this length at most, instead
# of the part's full bounding-box diagonal -- plenty to see whether a ray clears the surrounding
# geometry near the joint without the view being dominated by rays drawn far out into empty space.
RAY_DRAW_CAP_MM = 100.0


def prompt_tilt_deg(default):
    """Asks for the manual work-angle tilt. The default is whatever weld_sequence.py's own
    MANUAL_TEST_TILT_DEG currently holds, so a value already set there carries over."""
    raw = input(f"Tilt the tested torch direction off the ideal bisector by this work angle in "
                f"degrees (sign picks the side, 0 = pure bisector) [{default:g}]: ").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"Couldn't parse that -- using default ({default:g}).")
        return float(default)


def component_spec(value):
    """A component to weld against: a component number, or 'auto' (detect it per weld line)."""
    text = str(value).strip().lower()
    return "auto" if text == "auto" else int(text)


# One weld line per docx line: comma-separated edge IDs, optionally followed by which component
# it is welded onto -- "70,71,72,73", "70,71,72,73:3", "70,71,72,73 -> 3", "93 : auto". The arrow
# may also be Word's autocorrected "->" (a single arrow character).
_WELD_LINE_RE = re.compile(
    r"(?P<ids>\d+(?:\s*,\s*\d+)*)(?:\s*(?::|->|=>|→)\s*(?P<comp>\d+|auto))?",
    re.IGNORECASE)


def parse_weld_lines_from_docx(docx_path):
    """Reads the plain paragraph text out of a .docx via its raw word/document.xml (stdlib
    zipfile + regex -- no python-docx dependency needed) and treats every line that matches
    _WELD_LINE_RE as one weld line: its candidate-edge ID group (the same format
    find_weld_lines_interactive() would otherwise ask the user to type by hand) plus, optionally,
    the component it is welded onto. Returns a list of {"edge_ids": [...], "other_component":
    int | "auto" | None}, where None means "not stated on that line -- use the run-wide default".
    Any other text in the document (titles, notes) is ignored."""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    text = re.sub(r"<w:p\b[^>]*>|<w:br\b[^>]*>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    weld_line_specs = []
    for raw_line in text.split("\n"):
        match = _WELD_LINE_RE.fullmatch(raw_line.strip())
        if not match:
            continue
        weld_line_specs.append({
            "edge_ids": [int(x) for x in match["ids"].replace(" ", "").split(",")],
            "other_component": component_spec(match["comp"]) if match["comp"] else None,
        })
    return weld_line_specs


def detect_other_component(chain, own_component, faces, face_groups, component_of_face, proximity_tol):
    """Picks the component (other than the weld line's own) that the whole chain lies closest to:
    each candidate is scored by the distance from its WORST-matching piece to its nearest face, so
    a component touching every piece beats one that only touches a single piece. Returns None if
    nothing is within proximity_tol, since guessing a component that isn't actually touching the
    seam would only produce a meaningless weld line."""
    n_components = max(component_of_face) + 1
    scored = []
    for comp in range(n_components):
        if comp == own_component:
            continue
        groups = component_face_groups(faces, face_groups, component_of_face, comp)
        if not groups:
            continue
        dists = [find_closest_face_groups(edge, groups, faces, top_k=1)[0][1] for edge, _ in chain]
        scored.append((max(dists), sum(dists) / len(dists), comp))
    scored.sort()

    if not scored or scored[0][0] > proximity_tol:
        print(f"  [Warning] No other component lies within {proximity_tol} mm of every piece of this "
              f"weld line -- can't auto-detect what it's welded onto, discarding.")
        return None

    touching = [comp for worst, _mean, comp in scored if worst <= proximity_tol]
    if len(touching) > 1:
        print(f"  [Note] Components {touching} all lie within {proximity_tol} mm of this weld line -- "
              f"chose {scored[0][2]} (closest). If that's wrong, name it in the docx, e.g. "
              f"'ids:{touching[1]}'.")
    return scored[0][2]


def resolve_assembly_chain_faces_auto(tagged_mesh, faces, face_groups, component_of_face,
                                       chain, candidate_edges, own_info, proximity_tol,
                                       parallel_tol_deg, other_component):
    """Same face-resolution logic as weld_line_finder.resolve_assembly_chain_faces, but with no
    terminal/GUI prompts and no routine per-piece narration: the component this weld line is
    welded onto is given (or, for 'auto', detected from proximity -- see detect_other_component)
    instead of asked, and whichever far-side face comes out spatially closest is kept
    automatically instead of opening a manual picker window. Only genuine problems (ambiguous or
    out-of-tolerance matches, unresolvable chains) are printed -- the routine "this piece resolved
    to this face" narration is left out to keep the terminal output down to a final summary, not
    a per-piece trace."""
    per_kind = [own_info[_edge_index_in(edge, candidate_edges)] for edge, _ in chain]

    comps = {info[2] for info in per_kind}
    if len(comps) != 1:
        print(f"  [Warning] This chain's pieces belong to different components {sorted(comps)} -- "
              f"not a single weld line, discarding.")
        return None
    own_component = comps.pop()

    n_components = max(component_of_face) + 1
    detected = other_component == "auto"
    if detected:
        other_component = detect_other_component(
            chain, own_component, faces, face_groups, component_of_face, proximity_tol)
        if other_component is None:
            return None
    elif not 0 <= other_component < n_components:
        print(f"  [Warning] Component {other_component} doesn't exist (this assembly has components "
              f"0-{n_components - 1}) -- can't resolve, discarding.")
        return None

    if own_component == other_component:
        print(f"  [Warning] This weld line's own component is {own_component}, same as the "
              f"component it's supposed to be welded onto ({other_component}) -- can't resolve, "
              f"discarding.")
        return None

    other_face_groups = component_face_groups(faces, face_groups, component_of_face, other_component)
    if not other_face_groups:
        print(f"  [Warning] Component {other_component} has no faces -- can't resolve this weld line.")
        return None

    per_piece_faces = []
    for (edge, _reversed), (kind, info, _comp) in zip(chain, per_kind):
        ranked = find_closest_face_groups(edge, other_face_groups, faces)
        best_group, best_dist = ranked[0]
        ambiguous = len(ranked) > 1 and (ranked[1][1] - best_dist) < max(0.1 * best_dist, 0.1)

        if best_dist > proximity_tol:
            print(f"  [Warning] Closest face on component {other_component} is {best_dist:.2f} mm "
                  f"away (tolerance {proximity_tol} mm) -- this seam doesn't touch that component, "
                  f"discarding. Check the component number in the docx (or use 'auto'), or raise "
                  f"--proximity_tol if the gap is genuine.")
            return None
        elif ambiguous:
            print(f"  [Note] Multiple faces on component {other_component} are similarly close "
                  f"(~{best_dist:.2f} mm) -- keeping the closest one (unattended run).")

        if kind == "naked":
            own_group = info
        else:
            ga, gb = info
            own_group, _eliminated, _angle_kept, _angle_eliminated = eliminate_parallel_own_face(
                edge, ga, gb, best_group, faces, parallel_tol_deg)

        per_piece_faces.append((list(own_group), list(best_group)))

    print(f"  Resolved: component {own_component} <-> component {other_component}"
          f"{' (auto-detected)' if detected else ''} ({len(per_piece_faces)} piece(s))")
    return per_piece_faces


def _print_and_capture_ranked_summary(results, own_other, optimal):
    """Runs weld_sequence.print_ranked_summary as normal (so it still shows on the terminal) while
    also capturing exactly what it printed, so that text can be written to a log file afterward
    without re-implementing the table formatting a second time."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_ranked_summary(results, own_other, optimal)
    text = buf.getvalue()
    print(text, end="")
    return text


def run_sequence_search_quiet(tagged_mesh, weld_lines, own_other, baseline_present, near_tol,
                               max_full_enum, n_rings=0, clearance_radius=5.0, ring_offset=None):
    """Same search as weld_sequence.run_sequence_search, but skips the full per-order report that
    function prints for every single candidate ordering -- only the final ranked-summary table
    (or, past max_full_enum, the one DP-found order) is shown, and that table's text is returned
    too so the caller can save it to a log file."""
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
        print(f"\nEvaluating all {n}! possible weld order(s)...")
        results = enumerate_all_sequences(evaluator, n)
        optimal = min(results, key=_score)
        summary_text = _print_and_capture_ranked_summary(results, own_other, optimal)
    else:
        print(f"\n{n} weld lines -> too many orderings to enumerate (limit is {max_full_enum}, set "
              f"with --max_full_enum). Finding the order that minimizes total blocked rays via an "
              f"exact subset search instead.")
        optimal = find_optimal_via_dp(evaluator, n)
        summary_text = _print_and_capture_ranked_summary([optimal], own_other, optimal)

    return optimal, evaluator, summary_text


def visualize_weld_line_accessibility_capped(tagged_mesh, chain, pts, bis_arr, near_tol=0.5,
                                              n_rings=0, clearance_radius=5.0, ring_offset=None,
                                              draw_cap=RAY_DRAW_CAP_MM):
    """Same ray-cast + render as weld_line_finder.visualize_weld_line_accessibility, but always
    draws rays capped at `draw_cap` mm and never asks about a second, shorter-range view -- one
    visualization per weld line, done."""
    max_dist = bbox_diagonal(tagged_mesh)
    (combined_accessible, center_accessible, center_hit_dists,
     ring_accessible, ring_hit_dists, ring_origins, ring_dirs) = cast_ray_bundle_with_clearance(
        tagged_mesh, pts, bis_arr, max_dist, near_tol, n_rings, clearance_radius, ring_offset=ring_offset)

    n_accessible = int(combined_accessible.sum())
    print(f"  {n_accessible}/{len(pts)} accessible, {len(pts) - n_accessible}/{len(pts)} obstructed "
          f"(rays drawn up to {draw_cap:.0f} mm)")

    _render_bisector_scene(tagged_mesh, chain, pts, bis_arr, center_accessible, center_hit_dists,
                            ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                            clearance_radius, max_dist, n_accessible, draw_cap=draw_cap)


def main():
    parser = argparse.ArgumentParser(
        description="Showcase pipeline: reads the weld lines described in a .docx (one line per "
                     "weld, as comma-separated candidate-edge IDs -- the same IDs weld_line_finder.py "
                     "shows and normally asks you to type by hand) instead of picking them "
                     "interactively, resolves each one against the component it is welded onto "
                     "(named per line in the docx, e.g. '70,71,72,73:3', else --other_component; "
                     "'auto' detects it from proximity), then runs the same weld-order "
                     "search as weld_sequence.py and visualizes the optimal sequence's ray casts "
                     "step by step.")
    parser.add_argument("--step", default=str(THIS_DIR / "Assem2.step"), help="Path to input STEP assembly file")
    parser.add_argument("--docx", default=str(THIS_DIR / "Assem2.docx"), help="Path to the .docx describing weld lines")
    parser.add_argument("--other_component", type=component_spec, default=0,
                         help="Default component a weld line is welded onto when its docx line doesn't "
                              "name one (e.g. '70,71,72,73:3'): a component number, or 'auto' to "
                              "detect it per weld line from whichever other component the seam touches")
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Points sampled per weld line (skips the interactive prompt if given)")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Ray start offset (mm)")
    parser.add_argument("--proximity_tol", type=float, default=2.0,
                         help="Max distance (mm) for a face on the other component to be accepted "
                              "as a weld line's far-side face")
    parser.add_argument("--parallel_tol_deg", type=float, default=20.0,
                         help="For a same-component corner edge, an own-side candidate face within "
                              "this many degrees of parallel to the resolved far-side face is treated "
                              "as the hidden flush-contact face and eliminated")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    parser.add_argument("--max_full_enum", type=int, default=DEFAULT_MAX_FULL_ENUM,
                         help="Max weld lines to brute-force enumerate every ordering for")
    parser.add_argument("--n_rings", type=int, default=None,
                         help="Clearance ring rays per point simulating a spherical tool radius, "
                              "used for every accessibility check (skips the interactive prompt if "
                              "given; 0 disables)")
    parser.add_argument("--clearance_radius", type=float, default=None,
                         help="Spherical tool radius in mm for the clearance ring rays "
                              "(only used together with --n_rings)")
    parser.add_argument("--ring_offset", type=float, default=None,
                         help="Forward start-point offset (mm) for clearance ring rays "
                              "(default: clearance_radius; only used together with --n_rings)")
    parser.add_argument("--tilt_deg", type=float, default=None,
                         help="Manual work-angle tilt (degrees) of every tested ray off the ideal "
                              "bisector, rotated about each point's own tangent -- applied to the "
                              "order search AND the step-by-step visualizations (skips the "
                              "interactive prompt if given; 0 = pure bisector)")
    parser.add_argument("--ray_draw_cap", type=float, default=RAY_DRAW_CAP_MM,
                         help="Max ray length drawn in the final per-step visualizations (mm)")
    parser.add_argument("--log_file", default=None,
                         help="Where to save the final ranked-summary table as text (default: "
                              "a timestamped file next to this script; pass '' to disable saving)")
    args = parser.parse_args()

    num_samples = args.num_samples if args.num_samples is not None else prompt_num_samples()

    if args.n_rings is not None:
        n_rings = args.n_rings
        clearance_radius = args.clearance_radius if args.clearance_radius is not None else 5.0
        ring_offset = args.ring_offset
    else:
        n_rings, clearance_radius, ring_offset = prompt_for_clearance_rings()

    tilt_deg = args.tilt_deg if args.tilt_deg is not None \
        else prompt_tilt_deg(weld_sequence.MANUAL_TEST_TILT_DEG)
    # weld_sequence._test_directions reads this module-level constant at call time, and is what
    # both SequenceEvaluator.cost (the order search) and the step visualizations below use to get
    # their ray directions -- setting it here is what keeps the search and the visuals agreeing
    # on the same tilt, exactly as when it is edited by hand in weld_sequence.py.
    weld_sequence.MANUAL_TEST_TILT_DEG = tilt_deg
    if tilt_deg != 0:
        print(f"Testing every ray tilted {tilt_deg:+g} deg (work angle) off the ideal bisector, in "
              f"both the order search and the visualizations.")

    print(f"\nParsing weld lines from '{args.docx}'...")
    weld_line_specs = parse_weld_lines_from_docx(args.docx)
    if not weld_line_specs:
        raise RuntimeError(f"No weld line edge-ID groups found in '{args.docx}'.")
    print(f"Found {len(weld_line_specs)} weld line(s) described in the docx:")
    for i, spec in enumerate(weld_line_specs):
        stated = spec["other_component"]
        onto = f"component {stated}" if stated is not None else f"default ({args.other_component})"
        print(f"  {i}: edge ID(s) {spec['edge_ids']} -> welded onto {onto}")

    print("\nLoading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    solids = list(topo.solids())
    if len(solids) <= 1:
        raise RuntimeError(
            "This script is assembly-only (2+ separate components) -- this STEP file has a "
            "single solid.")

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
        raise RuntimeError("No usable edges were found -- nothing to analyze.")

    print("\nShowing the numbered candidate edges the docx's IDs refer to (close the window to continue)...")
    show_labeled_edges(tagged_mesh, candidate_edges)

    weld_lines = []
    for i, spec in enumerate(weld_line_specs):
        edge_ids = spec["edge_ids"]
        other_component = spec["other_component"] if spec["other_component"] is not None \
            else args.other_component
        print(f"\n--- Weld line {i} from docx: edge ID(s) {edge_ids} ---")
        if any(eid < 0 or eid >= len(candidate_edges) for eid in edge_ids):
            print(f"  [Warning] Edge ID(s) out of range (0-{len(candidate_edges) - 1}) -- skipped.")
            continue
        if len(set(edge_ids)) != len(edge_ids):
            print("  [Warning] Duplicate edge ID(s) on this line -- skipped.")
            continue

        chains = build_chains_from_ids(edge_ids, candidate_edges)
        if len(chains) > 1:
            print(f"  [Warning] These edge ID(s) don't form a single continuous line -- they split "
                  f"into {len(chains)} disconnected piece(s) -- skipped.")
            continue
        chain = chains[0]

        per_piece_faces = resolve_assembly_chain_faces_auto(
            tagged_mesh, faces, face_groups, component_of_face, chain, candidate_edges, own_info,
            args.proximity_tol, args.parallel_tol_deg, other_component)
        if per_piece_faces is None:
            print("  Could not resolve this weld line's faces -- skipped.")
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            pts, n1_arr, n2_arr, bis_arr, tan_arr = sample_and_analyze_chain(
                tagged_mesh, chain, per_piece_faces, faces, num_samples=num_samples)
        print(f"  Sampled {len(pts)} point(s) along the seam.")
        preview_weld_line(tagged_mesh, chain, per_piece_faces, pts, n1_arr, n2_arr, bis_arr)

        weld_lines.append({
            "docx_line": i, "edge_ids": edge_ids,
            "chain": chain, "per_piece_faces": per_piece_faces,
            "points": pts, "normals1": n1_arr, "normals2": n2_arr, "bisectors": bis_arr,
            "tangents": tan_arr,
        })

    if not weld_lines:
        print("\nNo weld lines could be resolved from the docx -- nothing to sequence.")
        return

    own_other = weld_own_other_components(weld_lines, component_of_face)
    print(f"\n{len(weld_lines)} weld line(s) auto-parsed and resolved:")
    weld_key_lines = [
        f"  {weld_label(i, own, other)}: docx line {weld_lines[i]['docx_line']}, edge ID(s) "
        f"{weld_lines[i]['edge_ids']}, {len(weld_lines[i]['points'])} sample point(s)"
        for i, (own, other) in enumerate(own_other)]
    print("\n".join(weld_key_lines))

    referenced = {c for pair in own_other for c in pair}
    baseline_present = set(range(n_components)) - referenced
    if baseline_present:
        print(f"[Setup] Component(s) {sorted(baseline_present)} aren't the target of any weld line "
              f"-- treated as fixed context, present for every step.")

    optimal, evaluator, summary_text = run_sequence_search_quiet(
        tagged_mesh, weld_lines, own_other, baseline_present, args.near_tol, args.max_full_enum,
        n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset)

    if args.log_file != "":
        log_path = Path(args.log_file) if args.log_file else \
            THIS_DIR / f"weld_sequence_results_{datetime.now():%Y%m%d_%H%M%S}.txt"
        header = (
            f"Weld sequence results -- {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"STEP file : {args.step}\n"
            f"Docx      : {args.docx}\n"
            f"Weld lines: {len(weld_lines)} (num_samples={num_samples}, n_rings={n_rings}"
            + (f", clearance_radius={clearance_radius}, ring_offset={ring_offset}" if n_rings > 0 else "")
            + f", tilt_deg={tilt_deg:g})\n"
            + "\n".join(weld_key_lines) + "\n\n"
        )
        log_path.write_text(header + summary_text, encoding="utf-8")
        print(f"\nFinal results table saved to '{log_path}'.")

    tilt_note = f", rays tilted {tilt_deg:+g} deg off the bisector" if tilt_deg != 0 else ""
    print(f"\nVisualizing the optimal sequence's ray casts, step by step "
          f"(rays capped at {args.ray_draw_cap:.0f} mm{tilt_note})...")
    for step in optimal["steps"]:
        wl = weld_lines[step["weld"]]
        print(f"\n--- {weld_label(step['weld'], *own_other[step['weld']])}: "
              f"present components {step['present']} ---")
        mesh = evaluator.obstruction_mesh(step["present"])
        visualize_weld_line_accessibility_capped(
            mesh, wl["chain"], wl["points"], _test_directions(wl), near_tol=args.near_tol,
            n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset,
            draw_cap=args.ray_draw_cap)


if __name__ == "__main__":
    main()
