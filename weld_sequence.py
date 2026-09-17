import argparse
from collections import deque

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    build_face_normal_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    pick_two_faces,
    find_shared_edges,
    assemble_edge_chains,
    sample_edge_with_bisector,
    compute_group_normal_signs_by_raycast,
    apply_normal_sign_correction,
    _id_to_face_index,
    _sample_edge_polyline,
    _sample_chain_polyline,
    bbox_diagonal,
)

COMPONENT_CMAP = "tab10"



def identify_components(shape):
    solids = list(TopologyExplorer(shape).solids())
    faces, component_of_face = [], []
    for comp_idx, solid in enumerate(solids):
        solid_faces = list(TopologyExplorer(solid).faces())
        faces.extend(solid_faces)
        component_of_face.extend([comp_idx] * len(solid_faces))
    return solids, faces, component_of_face


def add_component_ids(tagged_mesh, component_of_face):
    face_ids = tagged_mesh.cell_data["FaceID"]
    lookup = np.array(component_of_face, dtype=np.int32)
    tagged_mesh.cell_data["ComponentID"] = lookup[face_ids]
    return tagged_mesh



def confirm_or_retry(pick_fn, describe=None, max_attempts=10):
    result = None
    for _attempt in range(max_attempts):
        result = pick_fn()
        desc = describe(result) if describe else str(result)
        raw = input(f"Confirm: {desc}? [Y/n]: ").strip().lower()
        if raw in ("", "y", "yes"):
            return result
        print("Retrying that selection...")
    print("Max retry attempts reached -- using the last selection.")
    return result


def pick_component(mesh, component_of_face, title, highlight_color="gold"):
    picked = []
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, scalars="ComponentID", cmap=COMPONENT_CMAP, show_edges=True,
                edge_color="dimgray", show_scalar_bar=False)
    pl.add_text(title, font_size=11, color="black")

    def on_pick(picked_sub):
        if picked_sub is None or picked_sub.n_cells == 0:
            return
        fid = int(picked_sub.cell_data["FaceID"][0])
        comp_idx = component_of_face[fid]
        picked.clear()
        picked.append(comp_idx)
        sub = mesh.extract_cells(mesh.cell_data["ComponentID"] == comp_idx)
        pl.add_mesh(sub, color=highlight_color, name="picked_component")
        print(f"Picked component #{comp_idx}")

    pl.enable_element_picking(callback=on_pick, mode="cell", left_clicking=True, show_message=False)
    pl.show()

    return picked[0] if picked else None


def build_edge_pick_index(solids, n_per_edge=12):
    all_edges = []
    sample_pts = []
    owner_of_sample = []
    for comp_idx, solid in enumerate(solids):
        for edge in TopologyExplorer(solid).edges():
            edge_idx = len(all_edges)
            all_edges.append((edge, comp_idx))
            poly = _sample_edge_polyline(edge, n=n_per_edge)
            sample_pts.append(poly)
            owner_of_sample.extend([edge_idx] * len(poly))
    sample_pts = np.vstack(sample_pts).astype(np.float64)
    tree = KDTree(sample_pts)
    return all_edges, tree, np.array(owner_of_sample, dtype=np.int32)


def pick_edge_interactive(mesh, all_edges, tree, owner_of_sample, title):
    picked = []
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, scalars="ComponentID", cmap=COMPONENT_CMAP, show_edges=True,
                edge_color="dimgray", opacity=0.85, show_scalar_bar=False)
    pl.add_text(title, font_size=11, color="black")

    def on_pick(*_args):
        point = pl.picked_point
        if point is None:
            return
        _, idx = tree.query(point)
        edge_idx = int(owner_of_sample[idx])
        picked.clear()
        picked.append(edge_idx)
        edge, comp_idx = all_edges[edge_idx]
        poly = _sample_edge_polyline(edge, n=30)
        pl.add_lines(poly, color="magenta", width=6, connected=True, name="picked_edge")
        print(f"Picked edge on component #{comp_idx}. Close the window to continue.")

    pl.enable_surface_point_picking(callback=on_pick, left_clicking=True, show_message=False)
    pl.show()

    if not picked:
        return None
    return all_edges[picked[0]]


def resolve_line_faces(mesh, faces, face_groups, edge, comp_solid):
    auto_faces = list(TopologyExplorer(comp_solid).faces_from_edge(edge))
    if len(auto_faces) == 2:
        print("(This edge already borders 2 faces of its own component -- "
              "click them again if that's the correct pair, or click faces from "
              "a different component if this is really an inter-component joint.)")
    print("Click the two faces adjacent to this weld line, then close the window.")
    face1, face2, idx1, idx2 = pick_two_faces(mesh, faces, face_groups, base_scalars="ComponentID")
    return face1, face2, idx1, idx2


def resolve_weld_line_chain(topo, edge, face1, face2):
    shared = find_shared_edges(topo, face1, face2)
    if not shared:
        return [(edge, False)]

    chains = assemble_edge_chains(shared)
    for chain, _is_closed in chains:
        if any(e.IsSame(edge) for e, _ in chain):
            return chain
    return [(edge, False)]


def _face_group_key(fid, face_groups):
    return tuple(sorted(face_groups.get(fid, [fid])))


def _shapes_within(shape_a, shape_b, tol):
    tool = BRepExtrema_DistShapeShape(shape_a, shape_b)
    return tool.IsDone() and tool.Value() < tol


def find_component_pair_contacts(comp_a_idx, comp_b_idx, component_of_face, faces, face_groups, tol=1e-3):
    faces_a = [i for i, c in enumerate(component_of_face) if c == comp_a_idx]
    faces_b = [i for i, c in enumerate(component_of_face) if c == comp_b_idx]

    groups_a = {_face_group_key(i, face_groups) for i in faces_a}
    groups_b = {_face_group_key(i, face_groups) for i in faces_b}

    contacts = []
    for group_a in groups_a:
        for group_b in groups_b:
            close = any(_shapes_within(faces[ia], faces[ib], tol) for ia in group_a for ib in group_b)
            if close:
                contacts.append((list(group_a), list(group_b)))
    return contacts


def extract_group_contact_chains(faces, group_a, group_b, comp_solid_a, tol=1e-3):
    contact_edges = []
    seen = set()
    for ia in group_a:
        for e in TopologyExplorer(comp_solid_a).edges_from_face(faces[ia]):
            if id(e) in seen:
                continue
            if any(_shapes_within(e, faces[ib], tol) for ib in group_b):
                contact_edges.append(e)
                seen.add(id(e))
    return assemble_edge_chains(contact_edges) if contact_edges else []


def scan_component_pair_weld_lines(faces, component_of_face, solids, face_groups, comp_a_idx, comp_b_idx, tol=1e-3):
    contacts = find_component_pair_contacts(comp_a_idx, comp_b_idx, component_of_face, faces, face_groups, tol)
    candidates = []
    for group_a, group_b in contacts:
        chains = extract_group_contact_chains(faces, group_a, group_b, solids[comp_a_idx], tol)
        for chain, _closed in chains:
            candidates.append({
                "chain": chain,
                "face1": [faces[i] for i in group_a], "face2": [faces[i] for i in group_b],
                "face1_idx": group_a, "face2_idx": group_b,
                "comp_a": comp_a_idx, "comp_b": comp_b_idx,
            })
    return candidates


def preview_and_select_segments(mesh, candidate_lines):
    palette = ["lime", "cyan", "yellow", "orange", "magenta", "red", "blue", "white"]
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, color="lightgray", opacity=0.5)
    for i, wl in enumerate(candidate_lines):
        poly = _sample_chain_polyline(wl["chain"])
        color = palette[i % len(palette)]
        pl.add_lines(poly, color=color, width=5, connected=True, label=f"[{i}]")
        mid = poly[len(poly) // 2]
        pl.add_point_labels([mid], [str(i)], font_size=18, text_color=color, shape=None)
    pl.add_legend()
    pl.add_title("Additional contact segments found -- note which to keep, then close the window",
                 font_size=11)
    pl.show()

    raw = input(f"Add which segments? [all/none/comma-separated indices 0-{len(candidate_lines) - 1}]: ").strip().lower()
    if raw in ("all", "a", ""):
        return list(range(len(candidate_lines)))
    if raw in ("none", "no", "n"):
        return []
    try:
        return sorted(set(int(x) for x in raw.replace(" ", "").split(",") if x != ""))
    except ValueError:
        print("Couldn't parse that -- adding none.")
        return []


def _add_weld_line(weld_lines, line_data, comp_a, comp_b, note_suffix=""):
    label = f"WeldLine{len(weld_lines) + 1}"
    line_data["label"] = label
    weld_lines.append(line_data)
    n_pieces = len(line_data["chain"])
    loop_note = f" ({n_pieces} edge piece(s), auto-assembled into a loop)" if n_pieces > 1 else ""
    print(f"{label} recorded: connects component {comp_a} <-> component {comp_b}{loop_note}{note_suffix}")


def collect_weld_lines(mesh, faces, face_groups, all_edges, tree, owner_of_sample, component_of_face, solids, topo):
    weld_lines = []
    while True:
        raw = input("\nAdd weld lines between two components? [Y/n to finish]: ").strip().lower()
        if raw in ("n", "no"):
            break

        comp_a = confirm_or_retry(
            lambda: pick_component(mesh, component_of_face,
                                    "Left-click any face of the FIRST component, then close the window.",
                                    highlight_color="gold"),
            describe=lambda c: f"component #{c}" if c is not None else "nothing picked")
        comp_b = confirm_or_retry(
            lambda: pick_component(mesh, component_of_face,
                                    "Left-click any face of the SECOND component, then close the window.",
                                    highlight_color="cyan"),
            describe=lambda c: f"component #{c}" if c is not None else "nothing picked")
        if comp_a is None or comp_b is None or comp_a == comp_b:
            print("Need two distinct components -- try again.")
            continue

        print(f"\nScanning for contact regions between component {comp_a} and component {comp_b}...")
        candidates = scan_component_pair_weld_lines(faces, component_of_face, solids, face_groups, comp_a, comp_b)
        print(f"Found {len(candidates)} candidate weld line(s).")

        if candidates:
            chosen = preview_and_select_segments(mesh, candidates)
            for k in chosen:
                _add_weld_line(weld_lines, candidates[k], comp_a, comp_b, note_suffix=" (auto-detected)")

        while True:
            raw2 = input(f"\nManually add another weld line between component {comp_a} "
                         f"and component {comp_b} (e.g. if the scan missed one)? [y/N]: ").strip().lower()
            if raw2 not in ("y", "yes"):
                break

            picked = confirm_or_retry(
                lambda: pick_edge_interactive(
                    mesh, all_edges, tree, owner_of_sample,
                    "Left-click near the weld line, then close the window."),
                describe=lambda p: f"edge on component #{p[1]}" if p else "nothing picked")
            if picked is None:
                print("No edge picked -- try again.")
                continue
            edge, edge_comp_idx = picked

            face1, face2, idx1, idx2 = confirm_or_retry(
                lambda: resolve_line_faces(mesh, faces, face_groups, edge, solids[edge_comp_idx]),
                describe=lambda r: f"faces {r[2]} <-> {r[3]}")

            chain = resolve_weld_line_chain(topo, edge, face1, face2)
            line_data = {
                "chain": chain,
                "face1": face1, "face2": face2,
                "face1_idx": idx1, "face2_idx": idx2,
                "comp_a": comp_a, "comp_b": comp_b,
            }
            _add_weld_line(weld_lines, line_data, comp_a, comp_b, note_suffix=" (manual)")

    return weld_lines



def compute_weld_line_rays(tagged_mesh, weld_line, num_samples):
    face1, face2 = weld_line["face1"], weld_line["face2"]
    idx1, idx2 = weld_line["face1_idx"], weld_line["face2_idx"]
    mesh_fallback1 = build_face_normal_mesh(tagged_mesh, idx1)
    mesh_fallback2 = build_face_normal_mesh(tagged_mesh, idx2)

    pts, n1_arr, n2_arr, bis_arr, tan_arr, owner1_ids, owner2_ids = sample_edge_with_bisector(
        weld_line["chain"], face1, face2, num_samples=num_samples,
        mesh_fallback1=mesh_fallback1, mesh_fallback2=mesh_fallback2)

    signs1 = compute_group_normal_signs_by_raycast(mesh_fallback1, pts, n1_arr, owner1_ids)
    signs2 = compute_group_normal_signs_by_raycast(mesh_fallback2, pts, n2_arr, owner2_ids)
    n1_arr = apply_normal_sign_correction(n1_arr, owner1_ids, signs1)
    n2_arr = apply_normal_sign_correction(n2_arr, owner2_ids, signs2)

    bis = n1_arr + n2_arr
    bis_norms = np.linalg.norm(bis, axis=1, keepdims=True)
    bis_norms[bis_norms < 1e-6] = 1.0
    bis_arr = bis / bis_norms

    weld_line["pts"] = pts
    weld_line["bisector"] = bis_arr
    return weld_line


def precompute_hits(tagged_mesh, weld_line, max_dist, near_tol=0.5):
    comp_ids = tagged_mesh.cell_data["ComponentID"]
    hit_lists = []
    for pt, d in zip(weld_line["pts"], weld_line["bisector"]):
        start = pt + d * near_tol
        end = pt + d * max_dist
        hit_pts, hit_cells = tagged_mesh.ray_trace(start, end, first_point=False)
        if len(hit_pts) == 0:
            hit_lists.append([])
            continue
        dists = np.linalg.norm(hit_pts - start, axis=1)
        order = np.argsort(dists)
        hits = [(float(dists[k]), int(comp_ids[hit_cells[k]])) for k in order]
        hit_lists.append(hits)
    weld_line["hit_lists"] = hit_lists
    return weld_line



def is_line_accessible(weld_line, installed_components):
    relevant = installed_components | {weld_line["comp_a"], weld_line["comp_b"]}
    for hits in weld_line["hit_lists"]:
        for _dist, comp_id in hits:
            if comp_id in relevant:
                return False
    return True


def compute_precedence_constraints(weld_lines, base_component):
    baseline_set = {base_component}
    always_infeasible = [wl["label"] for wl in weld_lines if not is_line_accessible(wl, baseline_set)]

    feasible_lines = [wl for wl in weld_lines if wl["label"] not in always_infeasible]
    constraints = []
    for wl_j in feasible_lines:
        for wl_i in feasible_lines:
            if wl_i is wl_j:
                continue
            installed_with_i_only = baseline_set | {wl_i["comp_a"], wl_i["comp_b"]}
            if not is_line_accessible(wl_j, installed_with_i_only):
                constraints.append((wl_j["label"], wl_i["label"]))

    return always_infeasible, constraints


def _topological_order(labels, constraints):
    graph = {l: set() for l in labels}
    indeg = {l: 0 for l in labels}
    for j, i in constraints:
        if j not in graph or i not in graph:
            continue
        if i not in graph[j]:
            graph[j].add(i)
            indeg[i] += 1

    queue = deque(sorted(l for l in labels if indeg[l] == 0))
    order = []
    indeg_work = dict(indeg)
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in sorted(graph[n]):
            indeg_work[m] -= 1
            if indeg_work[m] == 0:
                queue.append(m)

    return order if len(order) == len(labels) else None


def backtracking_search(weld_lines, base_component):
    order = []

    def dfs(installed, remaining):
        if not remaining:
            return list(order)
        for k in remaining:
            wl = weld_lines[k]
            if is_line_accessible(wl, installed):
                order.append(wl["label"])
                rest = [r for r in remaining if r != k]
                result = dfs(installed | {wl["comp_a"], wl["comp_b"]}, rest)
                if result is not None:
                    return result
                order.pop()
        return None

    result = dfs({base_component}, list(range(len(weld_lines))))
    return result, result is not None


def find_valid_sequence(weld_lines, constraints, base_component, feasible_labels):
    by_label = {wl["label"]: wl for wl in weld_lines}
    feasible_lines = [by_label[l] for l in feasible_labels]

    order = _topological_order(feasible_labels, constraints)
    if order is not None:
        installed = {base_component}
        valid = True
        for label in order:
            wl = by_label[label]
            if not is_line_accessible(wl, installed):
                valid = False
                break
            installed |= {wl["comp_a"], wl["comp_b"]}
        if valid:
            return order, True

    return backtracking_search(feasible_lines, base_component)



def report_sequence(weld_lines, base_component_idx, always_infeasible, constraints, sequence, ok):
    print("\n" + "=" * 70)
    print("WELD SEQUENCE PLANNING SUMMARY")
    print("=" * 70)
    print(f"Base component            : #{base_component_idx}")
    print(f"Weld lines                : {len(weld_lines)}")

    if always_infeasible:
        print(f"\nAlways obstructed (no sequence position fixes these): {always_infeasible}")

    if constraints:
        print("\nPrecedence constraints found:")
        for j, i in constraints:
            print(f"  '{j}' must be welded before '{i}'")
    else:
        print("\nNo precedence constraints -- the feasible lines are order-independent.")

    if ok:
        print("\nExample valid sequence:")
        for idx, label in enumerate(sequence, 1):
            print(f"  {idx}. {label}")
    else:
        print("\nNo fully valid sequence found (conflicting/cyclic constraints).")
    print("=" * 70 + "\n")


def visualize_sequence(tagged_mesh, weld_lines, sequence):
    by_label = {wl["label"]: wl for wl in weld_lines}
    palette = ["red", "orange", "gold", "green", "blue", "purple", "magenta", "cyan", "lime", "brown"]

    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(tagged_mesh, color="lightgray", opacity=0.45)

    for i, label in enumerate(sequence):
        wl = by_label[label]
        poly = _sample_chain_polyline(wl["chain"])
        color = palette[i % len(palette)]
        pl.add_lines(poly, color=color, width=6, connected=True, label=f"{i + 1}. {label}")
        mid = poly[len(poly) // 2]
        pl.add_point_labels([mid], [str(i + 1)], font_size=20, text_color=color, shape=None)

    pl.add_legend()
    pl.add_title("Weld Sequence (numbered in welding order)", font_size=12)
    pl.show()



def main():
    parser = argparse.ArgumentParser(description="Weld sequence planning for STEP assemblies")
    parser.add_argument("--step", required=True, help="Path to input STEP assembly file")
    parser.add_argument("--num_samples", type=int, default=9, help="Sample points per weld line")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Near-field start offset in mm")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    args = parser.parse_args()

    print("Loading STEP assembly...")
    shape = load_step(args.step)
    solids, faces, component_of_face = identify_components(shape)
    print(f"Identified {len(solids)} component(s), {len(faces)} face(s) total.")
    if len(solids) < 2:
        raise RuntimeError(
            "This STEP file has only one solid component -- weld sequence planning needs an "
            "assembly of at least two separate components. Use weldfinal.py or "
            "weldfinal_assembly.py for a single weld joint instead."
        )

    face_groups = find_cylinder_face_groups(faces)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)
    add_component_ids(tagged_mesh, component_of_face)

    topo = TopologyExplorer(shape)

    print("\nOpening interactive window: left-click any face of the BASE component.")
    base_component_idx = confirm_or_retry(
        lambda: pick_component(tagged_mesh, component_of_face,
                                "Left-click any face of the BASE component, then close the window."),
        describe=lambda c: f"component #{c}" if c is not None else "nothing picked")
    if base_component_idx is None:
        raise RuntimeError("No base component was selected. Re-run and pick one.")

    all_edges, tree, owner_of_sample = build_edge_pick_index(solids)

    weld_lines = collect_weld_lines(
        tagged_mesh, faces, face_groups, all_edges, tree, owner_of_sample, component_of_face, solids, topo)
    if not weld_lines:
        print("No weld lines were selected. Exiting.")
        return

    max_dist = bbox_diagonal(tagged_mesh)
    print(f"\nSampling and ray-casting {len(weld_lines)} weld line(s)...")
    for wl in weld_lines:
        compute_weld_line_rays(tagged_mesh, wl, args.num_samples)
        precompute_hits(tagged_mesh, wl, max_dist, args.near_tol)

    always_infeasible, constraints = compute_precedence_constraints(weld_lines, base_component_idx)
    feasible_labels = [wl["label"] for wl in weld_lines if wl["label"] not in always_infeasible]
    sequence, ok = find_valid_sequence(weld_lines, constraints, base_component_idx, feasible_labels)

    report_sequence(weld_lines, base_component_idx, always_infeasible, constraints, sequence, ok)

    if ok:
        visualize_sequence(tagged_mesh, weld_lines, sequence)


if __name__ == '__main__':
    main()
