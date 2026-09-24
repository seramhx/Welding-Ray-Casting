import argparse

import numpy as np
import pyvista as pv

from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    assemble_edge_chains,
    _sample_edge_polyline,
    _sample_chain_polyline,
    _edge_length_estimate,
    _surface_point_and_normal,
    sample_edge_with_bisector,
    bbox_diagonal,
    prompt_for_clearance_rings,
    cast_ray_bundle_with_clearance,
    _render_bisector_scene,
    prompt_for_second_view_draw_distance,
    add_scene_mesh,
)


def find_face_index(face, faces):
    for i, f in enumerate(faces):
        if f.IsSame(face):
            return i
    return None


def identify_components(shape):
    """Splits an assembly into its separate solids and returns one flat, global face list
    (the index into it is used everywhere else in this script) plus which component each
    face belongs to."""
    solids = list(TopologyExplorer(shape).solids())
    faces, component_of_face = [], []
    for comp_idx, solid in enumerate(solids):
        solid_faces = list(TopologyExplorer(solid).faces())
        faces.extend(solid_faces)
        component_of_face.extend([comp_idx] * len(solid_faces))
    return solids, faces, component_of_face


def collect_candidate_edges(shape, faces, face_groups):
    """Single-body mode: every edge that separates two distinct faces (or face-groups) of the
    one solid -- both sides of the weld are already known from real B-Rep topology. Outer-
    boundary edges (only one adjacent face -- an open/degenerate parametrization edge) and the
    internal seams of a split-cylinder face group (an artifact of how STEP stored the surface,
    not a real corner) are excluded."""
    topo = TopologyExplorer(shape)
    edges = list(topo.edges())

    candidates, face_pairs = [], []
    n_boundary, n_artifact = 0, 0
    for e in edges:
        idxs = []
        for f in topo.faces_from_edge(e):
            i = find_face_index(f, faces)
            if i is not None and i not in idxs:
                idxs.append(i)

        if len(idxs) < 2:
            n_boundary += 1
            continue

        a, b = idxs[0], idxs[1]
        ga = tuple(sorted(face_groups.get(a, [a])))
        gb = tuple(sorted(face_groups.get(b, [b])))
        if ga == gb:
            n_artifact += 1
            continue

        candidates.append(e)
        face_pairs.append((ga, gb))

    print(f"[Setup] {len(edges)} total edge(s) on the part: {len(candidates)} usable as weld-line "
          f"candidate(s), {n_boundary} outer-boundary edge(s) skipped, "
          f"{n_artifact} split-surface artifact edge(s) skipped.")
    return candidates, face_pairs


def collect_candidate_edges_assembly(shape, faces, face_groups, component_of_face):
    """Assembly mode applies NO weld-line filtering/heuristic -- every real edge of every
    component is selectable, exactly like single-body mode; the user picks whichever edge(s)
    they want. What differs is how each edge's SECOND face gets resolved later:

    - If the edge already borders two faces of its own component (an ordinary internal corner,
      e.g. any edge of a plain box), both are already known from real topology -- exactly like
      single-body mode, no further lookup needed ("corner" kind).
    - If it borders only one face of its own component (a naked/open boundary edge of that part,
      e.g. a tube's open rim), the second face lies on a *different*, non-topologically-connected
      component and can only be resolved once the user says which component it's on, via spatial
      proximity ("naked" kind).

    Only truly degenerate edges are excluded: ones with no identifiable owning face at all, and
    the internal seams of a split-cylinder face group (an artifact of how STEP stored the
    surface, not a real corner)."""
    topo = TopologyExplorer(shape)
    edges = list(topo.edges())

    candidates, own_info = [], []
    n_unowned, n_artifact = 0, 0
    for e in edges:
        idxs = []
        for f in topo.faces_from_edge(e):
            i = find_face_index(f, faces)
            if i is not None and i not in idxs:
                idxs.append(i)

        if len(idxs) == 0:
            n_unowned += 1
            continue

        own_component = component_of_face[idxs[0]]

        if len(idxs) == 1:
            own_group = tuple(sorted(face_groups.get(idxs[0], [idxs[0]])))
            candidates.append(e)
            own_info.append(("naked", own_group, own_component))
        else:
            a, b = idxs[0], idxs[1]
            ga = tuple(sorted(face_groups.get(a, [a])))
            gb = tuple(sorted(face_groups.get(b, [b])))
            if ga == gb:
                n_artifact += 1
                continue
            candidates.append(e)
            own_info.append(("corner", (ga, gb), own_component))

    n_naked = sum(1 for kind, _, _ in own_info if kind == "naked")
    n_corner = sum(1 for kind, _, _ in own_info if kind == "corner")
    print(f"[Setup] {len(edges)} total edge(s) across all component(s): {len(candidates)} selectable "
          f"({n_naked} open-boundary edge(s) that will need a cross-component face picked, "
          f"{n_corner} same-component corner edge(s) with both faces already known), "
          f"{n_unowned} unowned edge(s) skipped, {n_artifact} split-surface artifact edge(s) skipped.")
    return candidates, own_info


def build_edge_line_mesh(edges, n_per_edge=25):
    all_points, lines = [], []
    offset = 0
    for edge in edges:
        poly = _sample_edge_polyline(edge, n_per_edge)
        n = len(poly)
        all_points.append(poly)
        lines.append(np.array([n] + list(range(offset, offset + n)), dtype=np.int64))
        offset += n

    pd = pv.PolyData()
    pd.points = np.vstack(all_points).astype(np.float32)
    pd.lines = np.concatenate(lines)
    pd.cell_data["EdgeID"] = np.arange(len(edges), dtype=np.int32)
    return pd


def show_labeled_edges(tagged_mesh, candidate_edges):
    """Draws the part with every candidate edge numbered, so the user can read off IDs and
    type them in the terminal instead of clicking (clicking thin edges directly is unreliable)."""
    edges_mesh = build_edge_line_mesh(candidate_edges)
    label_points = [_sample_edge_polyline(edge, 20)[10] for edge in candidate_edges]

    pl = pv.Plotter(window_size=[1200, 900])
    add_scene_mesh(pl, tagged_mesh, opacity=0.45)
    pl.add_mesh(edges_mesh, scalars="EdgeID", cmap="tab20", line_width=4,
                show_scalar_bar=False, render_lines_as_tubes=True)
    pl.add_point_labels(
        label_points, [str(i) for i in range(len(candidate_edges))],
        font_size=14, text_color="black", shape_color="white", shape_opacity=0.75,
        always_visible=True)
    pl.add_title(
        f"{len(candidate_edges)} candidate weld-line edges, numbered 0-{len(candidate_edges) - 1}\n"
        "Close this window, then enter edge ID(s) in the terminal.",
        font_size=11)
    pl.show()


def show_labeled_components(tagged_mesh, n_components, exclude=None):
    """Same idea as show_labeled_edges, but for whole components -- used to ask which OTHER
    part the weld line's far-side face belongs to, since assembly components don't share
    topology and so can't be resolved automatically the way single-body faces are."""
    pl = pv.Plotter(window_size=[1200, 900])
    pl.add_mesh(tagged_mesh, scalars="ComponentID", cmap="tab10", show_edges=True,
                edge_color="dimgray", show_scalar_bar=False)

    label_points, label_text = [], []
    for comp_idx in range(n_components):
        mask = tagged_mesh.cell_data["ComponentID"] == comp_idx
        if not np.any(mask):
            continue
        label_points.append(tagged_mesh.extract_cells(mask).points.mean(axis=0))
        label_text.append(str(comp_idx))

    pl.add_point_labels(label_points, label_text, font_size=20, text_color="black",
                         shape_color="white", shape_opacity=0.8, always_visible=True)
    own_note = f" (component {exclude} is this weld line's own part -- pick a different one)" if exclude is not None else ""
    pl.add_title(
        f"{n_components} component(s), numbered{own_note}\n"
        "Close this window, then enter the component ID in the terminal.",
        font_size=11)
    pl.show()


def prompt_for_component_id(n_components, exclude=None):
    while True:
        raw = input(f"Enter the component ID (0-{n_components - 1}) that the weld line's OTHER "
                    f"face belongs to: ").strip()
        if not raw.isdigit():
            print("Enter a number.")
            continue
        comp = int(raw)
        if not (0 <= comp < n_components):
            print(f"Out of range -- must be 0-{n_components - 1}.")
            continue
        if exclude is not None and comp == exclude:
            print("That's this weld line's own component -- pick a different one.")
            continue
        return comp


def parse_edge_id_list(raw, n_candidates):
    try:
        ids = [int(x) for x in raw.replace(" ", "").split(",") if x != ""]
    except ValueError:
        return None
    if not ids or len(set(ids)) != len(ids):
        return None
    if any(i < 0 or i >= n_candidates for i in ids):
        return None
    return ids


def build_chains_from_ids(edge_ids, candidate_edges):
    picked_edges = [candidate_edges[i] for i in edge_ids]
    chains = assemble_edge_chains(picked_edges)
    return [chain for chain, _is_closed in chains]


def _edge_index_in(edge, candidate_edges):
    return next(i for i, ce in enumerate(candidate_edges) if ce.IsSame(edge))


def _edge_face_distance(edge, face):
    tool = BRepExtrema_DistShapeShape(edge, face)
    return tool.Value() if tool.IsDone() else float("inf")


def component_face_groups(faces, face_groups, component_of_face, comp_idx):
    comp_faces = [i for i, c in enumerate(component_of_face) if c == comp_idx]
    seen, groups = set(), []
    for i in comp_faces:
        key = tuple(sorted(face_groups.get(i, [i])))
        if key not in seen:
            seen.add(key)
            groups.append(key)
    return groups


def find_closest_face_groups(edge, candidate_groups, faces, top_k=5):
    scored = [(grp, min(_edge_face_distance(edge, faces[i]) for i in grp)) for grp in candidate_groups]
    scored.sort(key=lambda item: item[1])
    return scored[:top_k]


def pick_face_group_by_id(tagged_mesh, ranked_groups, title):
    palette = ["lime", "cyan", "yellow", "orange", "magenta", "red", "blue", "white"]
    pl = pv.Plotter(window_size=[1100, 850])
    add_scene_mesh(pl, tagged_mesh, opacity=0.4)
    for i, (grp, dist) in enumerate(ranked_groups):
        sub = tagged_mesh.extract_cells(np.isin(tagged_mesh.cell_data["FaceID"], grp))
        color = palette[i % len(palette)]
        pl.add_mesh(sub, color=color, label=f"[{i}] face(s) {list(grp)} ({dist:.2f} mm)")
    pl.add_legend()
    pl.add_title(title, font_size=11)
    pl.show()

    while True:
        raw = input(f"Enter index of the correct far-side face [0-{len(ranked_groups) - 1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(ranked_groups):
            return ranked_groups[int(raw)][0]
        print("Invalid index, try again.")


def _local_face_normal(face_group, faces, gp_p):
    """The exact analytic surface normal of a face group AT a specific 3D point -- never a
    mesh-wide average. Averaging cell normals across an entire face is meaningless whenever the
    face wraps all the way around (e.g. a full 360 deg cylinder, or a group of several faces
    doing so together): normals point outward in every radial direction and largely cancel, so
    the 'average' collapses to a near-zero vector dominated by mesh discretization noise --
    verified directly on this project's own test geometry (magnitude ~0.0036 before
    renormalizing back to a unit vector, i.e. an essentially arbitrary direction). Evaluating the
    surface exactly at the point that actually matters (the edge itself) has no such failure
    mode, on a plane, a cylinder, or anything else."""
    face = faces[face_group[0]]
    _, vec, _ = _surface_point_and_normal(face, None, gp_p)
    return vec


def eliminate_parallel_own_face(edge, ga, gb, other_group, faces, parallel_tol_deg):
    """A 'corner' piece has two own-side face candidates (both real, both on the same
    component) but only one is actually exposed to the weld -- the other is the flush contact
    face against the other component (e.g. a box's own bottom face where it rests on a plate),
    identifiable because its normal is approximately PARALLEL (or anti-parallel) to the other
    component's resolved face normal, unlike a real fillet joint's two faces which meet at an
    angle (typically perpendicular). All three normals are evaluated at the edge's own midpoint,
    not averaged over their whole faces (see _local_face_normal)."""
    mid = _sample_edge_polyline(edge, 3)[1]
    gp_p = gp_Pnt(float(mid[0]), float(mid[1]), float(mid[2]))

    n_a = _local_face_normal(ga, faces, gp_p)
    n_b = _local_face_normal(gb, faces, gp_p)
    n_other = _local_face_normal(other_group, faces, gp_p)

    def angle_to_other(n):
        cos_a = np.clip(abs(np.dot(n, n_other)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_a))

    angle_a, angle_b = angle_to_other(n_a), angle_to_other(n_b)
    a_is_parallel = angle_a < parallel_tol_deg
    b_is_parallel = angle_b < parallel_tol_deg

    if a_is_parallel and not b_is_parallel:
        return gb, ga, angle_b, angle_a
    if b_is_parallel and not a_is_parallel:
        return ga, gb, angle_a, angle_b

    print(f"  [Note] Parallel-normal test didn't cleanly pick one face (angles to far face: "
          f"{angle_a:.1f} deg vs {angle_b:.1f} deg) -- keeping the more perpendicular one.")
    return (ga, gb, angle_a, angle_b) if angle_a >= angle_b else (gb, ga, angle_b, angle_a)


def resolve_assembly_chain_faces(tagged_mesh, faces, face_groups, component_of_face, n_components,
                                  chain, candidate_edges, own_info, proximity_tol, parallel_tol_deg=20.0):
    """Resolves both faces for every piece of a picked (unfiltered) chain. Every piece needs the
    OTHER component's face resolved by proximity (assembly components share no B-Rep topology),
    asked for ONCE per weld line, not per piece:

    - A 'naked' piece already has exactly one own-side face (a real open boundary edge) -- used
      directly.
    - A 'corner' piece has TWO own-side face candidates from real topology (an ordinary internal
      corner of its component, e.g. any edge of a plain box) -- both are stored temporarily, then
      whichever one is approximately parallel to the resolved other-component face (the flush,
      hidden contact face, e.g. the box's own bottom face against a plate) is eliminated, keeping
      the other (typically perpendicular -- a real exposed weld corner)."""
    per_kind = [own_info[_edge_index_in(edge, candidate_edges)] for edge, _ in chain]

    comps = {info[2] for info in per_kind}
    if len(comps) != 1:
        print(f"[Warning] This chain's pieces belong to different components {sorted(comps)} -- "
              f"not a single weld line, discarding.")
        return None
    own_component = comps.pop()

    print(f"This weld line belongs to component {own_component}.")
    show_labeled_components(tagged_mesh, n_components, exclude=own_component)
    other_comp = prompt_for_component_id(n_components, exclude=own_component)
    other_face_groups = component_face_groups(faces, face_groups, component_of_face, other_comp)
    if not other_face_groups:
        print(f"[Warning] Component {other_comp} has no faces -- can't resolve this weld line.")
        return None

    per_piece_faces = []
    for (edge, _reversed), (kind, info, _comp) in zip(chain, per_kind):
        ranked = find_closest_face_groups(edge, other_face_groups, faces)
        best_group, best_dist = ranked[0]
        ambiguous = len(ranked) > 1 and (ranked[1][1] - best_dist) < max(0.1 * best_dist, 0.1)

        if best_dist > proximity_tol:
            print(f"  [Warning] Closest face on component {other_comp} is {best_dist:.2f} mm away "
                  f"(tolerance is {proximity_tol} mm) -- pick the correct far-side face manually.")
            best_group = pick_face_group_by_id(
                tagged_mesh, ranked,
                f"No face within tolerance -- pick the far-side face on component {other_comp}")
        elif ambiguous:
            print(f"  [Note] Multiple faces on component {other_comp} are similarly close "
                  f"(~{best_dist:.2f} mm) -- pick the correct one.")
            best_group = pick_face_group_by_id(
                tagged_mesh, ranked,
                f"Multiple close candidates -- pick the far-side face on component {other_comp}")
        else:
            print(f"  far-side face resolved: face(s) {list(best_group)} (component {other_comp}), "
                  f"{best_dist:.3f} mm from this piece")

        if kind == "naked":
            own_group = info
            print(f"  piece resolved: face(s) {list(own_group)} (component {own_component}) <-> "
                  f"face(s) {list(best_group)} (component {other_comp})")
        else:
            ga, gb = info
            own_group, eliminated, angle_kept, angle_eliminated = eliminate_parallel_own_face(
                edge, ga, gb, best_group, faces, parallel_tol_deg)
            print(f"  piece resolved: face(s) {list(own_group)} (component {own_component}, kept -- "
                  f"{angle_kept:.1f} deg from far face) <-> face(s) {list(best_group)} (component "
                  f"{other_comp}); eliminated face(s) {list(eliminated)} ({angle_eliminated:.1f} deg -- "
                  f"parallel/flush contact)")

        per_piece_faces.append((list(own_group), list(best_group)))

    return per_piece_faces


def _allocate_samples_by_length(chain, num_samples, min_per_piece=2):
    """Splits num_samples across the chain's pieces in proportion to each piece's real arc
    length (_edge_length_estimate walks the actual 3D curve regardless of type -- line, arc,
    spline -- so a tightly curved piece isn't under-counted just because it spans a short
    parameter range). Independently rounding each piece's share (e.g. round(num_samples *
    piece_len / total_len)) does not reliably reproduce num_samples as the total: with 4 equal-
    length pieces and num_samples=15, each share is exactly 3.75, and every piece rounds to 4,
    giving 16, not 15. This uses the largest-remainder (Hamilton) apportionment method instead --
    the same style of method used to fairly divide fixed seats by population -- which always
    distributes exactly num_samples as long as no per-piece minimum had to be enforced."""
    lengths = [_edge_length_estimate(edge) for edge, _ in chain]
    total_len = sum(lengths) or 1.0
    n = len(lengths)
    target = max(num_samples, min_per_piece * n)

    exact = [target * length / total_len for length in lengths]
    base = [int(e) for e in exact]
    remainder = target - sum(base)

    order = sorted(range(n), key=lambda i: exact[i] - base[i], reverse=True)
    for i in range(remainder):
        base[order[i % n]] += 1

    return [max(min_per_piece, b) for b in base]


def sample_and_analyze_chain(tagged_mesh, chain, per_piece_faces, faces, num_samples=15):
    """Samples points along the weld line and extracts normals per EDGE PIECE (each piece's own
    two adjacent faces are supplied via per_piece_faces, resolved upstream either by topology --
    single-body mode -- or by proximity across components -- assembly mode). A single continuous
    weld line can cross a corner and border a different 'other' face on each piece (e.g. a line
    running around three sides of a boss touches the same base-plate face the whole way, but a
    different boss side-face per piece), so faces are never assumed constant across the chain.

    Normals come from sample_edge_with_bisector's own resolution order: the exact analytic
    surface at that point if the edge truly, topologically owns that face (pcurve lookup); pure
    mesh-facet fallback is deliberately NOT used here. A cross-component 'other' face (assembly
    mode) never has a real pcurve on that face -- the edge doesn't belong to its topology at all
    -- so it always falls through to sample_edge_with_bisector's exact-projection fallback
    (ShapeAnalysis_Surface, projecting the point onto the face's true analytic surface). A
    mesh-facet nearest-neighbor fallback was tried here previously and rejected: for a curved
    (e.g. cylindrical) other face, snapping to the nearest MESH TRIANGLE is only an approximation
    of the surface and can snap to a facet that isn't the true local point, giving normals that
    barely change along the seam (near-parallel) instead of properly sweeping with the curve."""
    piece_sample_counts = _allocate_samples_by_length(chain, num_samples)

    all_pts, all_n1, all_n2, all_bis, all_tan, all_o1, all_o2 = [], [], [], [], [], [], []
    for (edge, reversed_), (face1_idx, face2_idx), piece_samples in zip(chain, per_piece_faces, piece_sample_counts):
        face1 = [faces[i] for i in face1_idx]
        face2 = [faces[i] for i in face2_idx]

        pts, n1, n2, bis, tan, o1, o2 = sample_edge_with_bisector(
            [(edge, reversed_)], face1, face2, num_samples=piece_samples)

        all_pts.append(pts); all_n1.append(n1); all_n2.append(n2)
        all_bis.append(bis); all_tan.append(tan)
        all_o1.extend(o1); all_o2.extend(o2)

    pts = np.concatenate(all_pts)
    n1_arr, n2_arr, bis_arr, tan_arr = (np.concatenate(a) for a in (all_n1, all_n2, all_bis, all_tan))

    joint_angles = np.degrees(np.arccos(np.clip(np.sum(n1_arr * n2_arr, axis=1), -1.0, 1.0)))
    print(f"  Weld line: {len(chain)} piece(s), {len(pts)} sample point(s) total, "
          f"face-normal angle min={joint_angles.min():.1f} max={joint_angles.max():.1f} "
          f"mean={joint_angles.mean():.1f} deg")
    for i, (f1, f2) in enumerate(per_piece_faces):
        print(f"    piece {i}: face(s) {f1} <-> face(s) {f2}")

    print("  [Debug] per-sample-point normal resolution:")
    for i in range(len(pts)):
        method1 = "pcurve(exact)" if all_o1[i] is not None else "projected(exact)"
        method2 = "pcurve(exact)" if all_o2[i] is not None else "projected(exact)"
        print(f"    [{i:>3}] pt=({pts[i][0]:.3f}, {pts[i][1]:.3f}, {pts[i][2]:.3f}) "
              f"n1=({n1_arr[i][0]:+.4f}, {n1_arr[i][1]:+.4f}, {n1_arr[i][2]:+.4f}) [{method1}]  "
              f"n2=({n2_arr[i][0]:+.4f}, {n2_arr[i][1]:+.4f}, {n2_arr[i][2]:+.4f}) [{method2}]  "
              f"angle={joint_angles[i]:.1f}deg  "
              f"bis=({bis_arr[i][0]:+.4f}, {bis_arr[i][1]:+.4f}, {bis_arr[i][2]:+.4f})")

    return pts, n1_arr, n2_arr, bis_arr, tan_arr


def resolve_single_body_per_piece_faces(chain, candidate_edges, face_pairs):
    per_piece = []
    for edge, _reversed in chain:
        idx = _edge_index_in(edge, candidate_edges)
        ga, gb = face_pairs[idx]
        per_piece.append((list(ga), list(gb)))
    return per_piece


def _common_and_varying_faces(per_piece_faces):
    all_used = set()
    for f1, f2 in per_piece_faces:
        all_used.update(f1); all_used.update(f2)
    common = set(per_piece_faces[0][0]) | set(per_piece_faces[0][1])
    for f1, f2 in per_piece_faces[1:]:
        common &= (set(f1) | set(f2))
    return sorted(common), sorted(all_used - common)


def preview_weld_line(tagged_mesh, chain, per_piece_faces, pts, n1_arr, n2_arr, bis_arr):
    arrow_len = 0.05 * bbox_diagonal(tagged_mesh)
    common_faces, varying_faces = _common_and_varying_faces(per_piece_faces)

    pl = pv.Plotter(window_size=[1100, 850])
    add_scene_mesh(pl, tagged_mesh, opacity=0.5)

    if common_faces:
        common_sub = tagged_mesh.extract_cells(np.isin(tagged_mesh.cell_data["FaceID"], common_faces))
        pl.add_mesh(common_sub, color="salmon", opacity=0.8, label=f"Shared face(s) {common_faces}")
    if varying_faces:
        varying_sub = tagged_mesh.extract_cells(np.isin(tagged_mesh.cell_data["FaceID"], varying_faces))
        pl.add_mesh(varying_sub, color="lightblue", opacity=0.8, label=f"Other face(s) {varying_faces}")

    poly = _sample_chain_polyline(chain)
    pl.add_lines(poly, color="black", width=6, connected=True, label="Weld line")

    pl.add_arrows(pts, n1_arr, mag=arrow_len, color="red", label="Face1 normal")
    pl.add_arrows(pts, n2_arr, mag=arrow_len, color="blue", label="Face2 normal")
    pl.add_arrows(pts, bis_arr, mag=arrow_len * 1.3, color="lime", label="Bisector (torch dir)")
    pl.add_points(pts, color="black", point_size=8, render_points_as_spheres=True)

    pl.add_legend()
    pl.add_title("Resolved weld line: normals + bisector -- close window to continue", font_size=11)
    pl.show()


def find_weld_lines_interactive(tagged_mesh, faces, candidate_edges, num_samples, resolve_faces_fn):
    weld_lines = []
    show_labeled_edges(tagged_mesh, candidate_edges)

    while True:
        raw = input(
            f"\n{len(weld_lines)} weld line(s) found so far.\n"
            f"Enter edge ID(s) for the next weld line -- a single ID, or comma-separated IDs if it's "
            f"several connected pieces forming ONE continuous line -- 'show' to redisplay the numbered "
            f"edges, or blank to finish: ").strip().lower()

        if raw == "":
            break
        if raw in ("show", "s"):
            show_labeled_edges(tagged_mesh, candidate_edges)
            continue

        edge_ids = parse_edge_id_list(raw, len(candidate_edges))
        if edge_ids is None:
            print(f"Couldn't parse that -- enter comma-separated edge IDs between 0 and "
                  f"{len(candidate_edges) - 1}, with no duplicates.")
            continue

        chains = build_chains_from_ids(edge_ids, candidate_edges)
        if len(chains) > 1:
            print(f"Those {len(edge_ids)} edge(s) don't form a single continuous line -- they split into "
                  f"{len(chains)} disconnected piece(s). Re-enter the IDs for just one continuous weld line.")
            continue
        chain = chains[0]

        per_piece_faces = resolve_faces_fn(chain)
        if per_piece_faces is None:
            print("Could not resolve this weld line's faces -- discarded.")
            continue

        pts, n1_arr, n2_arr, bis_arr, tan_arr = sample_and_analyze_chain(
            tagged_mesh, chain, per_piece_faces, faces, num_samples=num_samples)
        preview_weld_line(tagged_mesh, chain, per_piece_faces, pts, n1_arr, n2_arr, bis_arr)

        if input("Keep this weld line? [Y/n]: ").strip().lower() in ("n", "no"):
            print("Discarded.")
            continue

        weld_lines.append({
            "chain": chain, "per_piece_faces": per_piece_faces,
            "points": pts, "normals1": n1_arr, "normals2": n2_arr, "bisectors": bis_arr,
            "tangents": tan_arr,
        })

    return weld_lines


def visualize_weld_line_accessibility(tagged_mesh, chain, pts, bis_arr, near_tol=0.5,
                                       n_rings=0, clearance_radius=5.0, ring_offset=None):
    """Ray-casts and visualizes an entire weld line (every piece, every sample point) as one
    group -- one accessibility summary and one 3D scene -- rather than one per edge piece, even
    though each piece may border a different 'other' face (resolved per-piece upstream)."""
    max_dist = bbox_diagonal(tagged_mesh)
    print(f"Casting rays toward the joint bisector for all {len(pts)} sample point(s) across "
          f"{len(chain)} piece(s), length = part bbox diagonal = {max_dist:.2f} mm")
    if n_rings > 0:
        effective_offset = ring_offset if ring_offset is not None else clearance_radius
        print(f"Simulating a spherical tool of radius {clearance_radius:.1f} mm via {n_rings} clearance "
              f"ring ray(s) per point (start offset {effective_offset:.1f} mm forward along the ray)")

    (combined_accessible, center_accessible, center_hit_dists,
     ring_accessible, ring_hit_dists, ring_origins, ring_dirs) = cast_ray_bundle_with_clearance(
        tagged_mesh, pts, bis_arr, max_dist, near_tol, n_rings, clearance_radius, ring_offset=ring_offset)

    n_accessible = int(combined_accessible.sum())
    print("\n" + "=" * 70)
    print("WELD LINE ACCESSIBILITY SUMMARY")
    print("=" * 70)
    print(f"Sample points along weld line : {len(pts)}")
    if n_rings > 0:
        print(f"Center-ray only accessible          : {int(center_accessible.sum())} / {len(pts)}")
        print(f"Accessible with {clearance_radius:.1f} mm tool clearance : {n_accessible} / {len(pts)}")
    else:
        print(f"Accessible points        : {n_accessible} / {len(pts)}")
    print(f"Obstructed points        : {len(pts) - n_accessible} / {len(pts)}")
    if int(center_accessible.sum()) < len(pts):
        obstructed_hits = center_hit_dists[~center_accessible]
        print(f"Obstructed hit distances : min={obstructed_hits.min():.3f} mm, "
              f"max={obstructed_hits.max():.3f} mm, mean={obstructed_hits.mean():.3f} mm "
              f"(near_tol={near_tol} mm)")
    print("=" * 70 + "\n")

    _render_bisector_scene(tagged_mesh, chain, pts, bis_arr, center_accessible, center_hit_dists,
                            ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                            clearance_radius, max_dist, n_accessible)

    draw_cap = prompt_for_second_view_draw_distance()
    if draw_cap is not None:
        _render_bisector_scene(tagged_mesh, chain, pts, bis_arr, center_accessible, center_hit_dists,
                                ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                                clearance_radius, max_dist, n_accessible, draw_cap=draw_cap)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively find weld lines on a STEP part or assembly by entering the IDs of "
                     "numbered edges (shown in a 3D view), then resolve the adjacent face(s) and "
                     "extract normals/bisector along the seam.")
    parser.add_argument("--step", required=True, help="Path to input STEP file (single solid or assembly)")
    parser.add_argument("--num_samples", type=int, default=15, help="Points sampled along each weld line")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Ray start offset (mm), used for --analyze")
    parser.add_argument("--proximity_tol", type=float, default=2.0,
                         help="Assembly mode only: max distance (mm) for a face on the chosen other "
                              "component to be accepted as the weld line's far-side face")
    parser.add_argument("--parallel_tol_deg", type=float, default=20.0,
                         help="Assembly mode only: for a same-component corner edge, an own-side "
                              "candidate face within this many degrees of parallel to the resolved "
                              "far-side face is treated as the hidden flush-contact face and eliminated")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    args = parser.parse_args()

    print("Loading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    solids = list(topo.solids())
    assembly = len(solids) > 1

    if assembly:
        solids, faces, component_of_face = identify_components(shape)
        print(f"Assembly mode: {len(solids)} component(s), {len(faces)} face(s) total.")
        face_groups = find_cylinder_face_groups(faces, component_of_face=component_of_face)
    else:
        faces = list(topo.faces())
        component_of_face = None
        print(f"Single-body mode: {len(faces)} face(s).")
        face_groups = find_cylinder_face_groups(faces)

    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)

    if assembly:
        lookup = np.array(component_of_face, dtype=np.int32)
        tagged_mesh.cell_data["ComponentID"] = lookup[tagged_mesh.cell_data["FaceID"]]

        candidate_edges, own_info = collect_candidate_edges_assembly(
            shape, faces, face_groups, component_of_face)
        if not candidate_edges:
            raise RuntimeError("No usable edges were found -- nothing to pick.")

        def resolve_faces_fn(chain):
            return resolve_assembly_chain_faces(
                tagged_mesh, faces, face_groups, component_of_face, len(solids),
                chain, candidate_edges, own_info, args.proximity_tol, args.parallel_tol_deg)
    else:
        candidate_edges, face_pairs = collect_candidate_edges(shape, faces, face_groups)
        if not candidate_edges:
            raise RuntimeError("No interior edges (separating two distinct faces) were found -- nothing to pick.")

        def resolve_faces_fn(chain):
            return resolve_single_body_per_piece_faces(chain, candidate_edges, face_pairs)

    weld_lines = find_weld_lines_interactive(
        tagged_mesh, faces, candidate_edges, args.num_samples, resolve_faces_fn)

    print(f"\nTotal weld lines identified: {len(weld_lines)}")
    for i, wl in enumerate(weld_lines, 1):
        pair_desc = ", ".join(f"{f1}<->{f2}" for f1, f2 in wl["per_piece_faces"])
        print(f"  {i}. {len(wl['chain'])} edge piece(s) [{pair_desc}], {len(wl['points'])} sample point(s)")

    if not weld_lines:
        return

    if input("\nRun full ray-cast accessibility analysis on these weld line(s) now? [y/N]: ").strip().lower() in ("y", "yes"):
        n_rings, clearance_radius, ring_offset = prompt_for_clearance_rings()
        for i, wl in enumerate(weld_lines, 1):
            print(f"\n--- Weld line {i}/{len(weld_lines)} ({len(wl['chain'])} piece(s)) ---")
            visualize_weld_line_accessibility(
                tagged_mesh, wl["chain"], wl["points"], wl["bisectors"], near_tol=args.near_tol,
                n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset)


if __name__ == '__main__':
    main()
