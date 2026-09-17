import argparse

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    pick_two_faces,
    assemble_edge_chains,
    _sample_edge_polyline,
    _sample_chain_polyline,
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


def visualize_components(mesh):
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, scalars="ComponentID", cmap=COMPONENT_CMAP, show_edges=True,
                edge_color="dimgray", show_scalar_bar=True)
    pl.add_title("Stage 0: Identified components (one color per component) -- close to continue",
                 font_size=12)
    pl.show()


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



def _dedup_edges_by_identity(edges):
    unique = []
    for e in edges:
        if not any(e.IsSame(u) for u in unique):
            unique.append(e)
    return unique


def repair_cylinder_group_loops(faces, face_groups):
    repaired = {}
    groups = sorted({tuple(sorted(g)) for g in face_groups.values() if len(g) > 1})
    for group in groups:
        per_face_edges = [_dedup_edges_by_identity(list(TopologyExplorer(faces[fid]).edges()))
                           for fid in group]
        unique_edges = _dedup_edges_by_identity([e for edges in per_face_edges for e in edges])

        boundary_edges = [
            e for e in unique_edges
            if sum(1 for edges in per_face_edges if any(e.IsSame(o) for o in edges)) == 1
        ]
        repaired[group] = assemble_edge_chains(boundary_edges)
    return repaired


def report_repaired_loops(group_loops):
    if not group_loops:
        print("[Stage 1] No split-cylinder face groups to repair.")
        return
    for group, chains in group_loops.items():
        summary = ", ".join(f"{'loop' if closed else 'open chain'} of {len(c)} piece(s)"
                             for c, closed in chains)
        print(f"[Stage 1] Group {group}: {len(chains)} edge unit(s) -- {summary}")


def visualize_repaired_loops(mesh, group_loops):
    if not group_loops:
        print("[Stage 1] Nothing to visualize -- no split-cylinder groups found.")
        return
    palette = ["red", "lime", "cyan", "orange", "magenta", "yellow", "blue", "white"]
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, color="lightgray", opacity=0.4)
    i = 0
    for group, chains in group_loops.items():
        for chain, is_closed in chains:
            poly = _sample_chain_polyline(chain)
            color = palette[i % len(palette)]
            tag = "loop" if is_closed else "open chain"
            pl.add_lines(poly, color=color, width=5, connected=True,
                         label=f"group {group}: {tag}, {len(chain)} piece(s)")
            mid = poly[len(poly) // 2]
            pl.add_point_labels([mid], [str(i)], font_size=16, text_color=color, shape=None)
            i += 1
    pl.add_legend()
    pl.add_title(f"Stage 1: Repaired split-cylinder edge loops ({i} found) -- close to continue",
                 font_size=12)
    pl.show()



def _face_group_key(fid, face_groups):
    return tuple(sorted(face_groups.get(fid, [fid])))


def _shapes_within(shape_a, shape_b, tol):
    tool = BRepExtrema_DistShapeShape(shape_a, shape_b)
    return tool.IsDone() and tool.Value() < tol


def build_component_point_trees(tagged_mesh, n_components):
    comp_ids = tagged_mesh.cell_data["ComponentID"]
    tris = tagged_mesh.faces.reshape(-1, 4)[:, 1:]
    trees = {}
    for comp_idx in range(n_components):
        point_ids = np.unique(tris[comp_ids == comp_idx])
        trees[comp_idx] = KDTree(tagged_mesh.points[point_ids])
    return trees


def chain_is_near_component(chain, comp_tree, tol, n_probe=6):
    n_per_edge = max(2, n_probe // max(1, len(chain)) + 1)
    poly = _sample_chain_polyline(chain, n_per_edge=n_per_edge)
    if len(poly) == 0:
        return False
    probe_idx = np.linspace(0, len(poly) - 1, min(n_probe, len(poly))).astype(int)
    probe_pts = poly[probe_idx]
    dists, _ = comp_tree.query(probe_pts)
    return bool(np.all(dists < tol))


def component_candidate_chains(comp_idx, component_of_face, faces, face_groups, group_loops):
    comp_faces = [i for i, c in enumerate(component_of_face) if c == comp_idx]
    seen_groups = set()
    seen_edges = []
    candidates = []

    for fid in comp_faces:
        group_key = _face_group_key(fid, face_groups)
        if len(group_key) > 1:
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            for chain, _closed in group_loops.get(group_key, []):
                candidates.append(chain)
                seen_edges.extend(e for e, _ in chain)
        else:
            for e in TopologyExplorer(faces[fid]).edges():
                if any(e.IsSame(s) for s in seen_edges):
                    continue
                candidates.append([(e, False)])
                seen_edges.append(e)
    return candidates


def scan_weld_line_candidates(comp_a, comp_b, component_of_face, faces, face_groups, group_loops,
                               comp_trees, tol):
    candidates = []
    for owner_comp, other_comp in [(comp_a, comp_b), (comp_b, comp_a)]:
        chains = component_candidate_chains(owner_comp, component_of_face, faces, face_groups, group_loops)
        for chain in chains:
            if chain_is_near_component(chain, comp_trees[other_comp], tol):
                candidates.append({"chain": chain, "owner_comp": owner_comp, "other_comp": other_comp})
    return candidates


def preview_and_select_candidates(mesh, candidates, title):
    palette = ["lime", "cyan", "yellow", "orange", "magenta", "red", "blue", "white"]
    pl = pv.Plotter(window_size=[1100, 850])
    pl.add_mesh(mesh, color="lightgray", opacity=0.5)
    for i, cand in enumerate(candidates):
        poly = _sample_chain_polyline(cand["chain"])
        color = palette[i % len(palette)]
        pl.add_lines(poly, color=color, width=5, connected=True, label=f"[{i}]")
        mid = poly[len(poly) // 2]
        pl.add_point_labels([mid], [str(i)], font_size=18, text_color=color, shape=None)
    pl.add_legend()
    pl.add_title(title, font_size=11)
    pl.show()

    raw = input(f"Use which candidate(s)? [all/none/comma-separated indices 0-{len(candidates) - 1}]: ").strip().lower()
    if raw in ("all", "a", ""):
        return list(range(len(candidates)))
    if raw in ("none", "no", "n"):
        return []
    try:
        return sorted(set(int(x) for x in raw.replace(" ", "").split(",") if x != ""))
    except ValueError:
        print("Couldn't parse that -- using none.")
        return []



def resolve_own_face(chain, faces, face_groups, component_of_face, owner_comp):
    edge0 = chain[0][0]
    comp_faces = [i for i, c in enumerate(component_of_face) if c == owner_comp]
    bordering = [i for i in comp_faces if any(e.IsSame(edge0) for e in TopologyExplorer(faces[i]).edges())]
    if not bordering:
        return [], []
    group = list(face_groups.get(bordering[0], bordering))
    return group, [faces[i] for i in group]


def find_far_side_faces(chain, faces, component_of_face, face_groups, other_comp, tol):
    other_faces = [i for i, c in enumerate(component_of_face) if c == other_comp]
    seen_groups = set()
    found = []
    for fid in other_faces:
        group_key = _face_group_key(fid, face_groups)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        if any(_shapes_within(e, faces[fid], tol) for e, _ in chain):
            found.append(list(group_key))
    return found


def pick_far_face_by_index(mesh, candidate_groups):
    palette = ["lime", "cyan", "yellow", "orange", "magenta", "red"]
    pl = pv.Plotter(window_size=[1000, 800])
    pl.add_mesh(mesh, color="lightgray", opacity=0.4)
    for i, group in enumerate(candidate_groups):
        sub = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], group))
        color = palette[i % len(palette)]
        pl.add_mesh(sub, color=color, label=f"[{i}] faces {group}")
    pl.add_legend()
    pl.add_title("Multiple far-side face candidates found -- note which to keep, then close the window",
                 font_size=11)
    pl.show()

    choice = -1
    while not (0 <= choice < len(candidate_groups)):
        raw = input(f"Enter index of the correct far-side face [0-{len(candidate_groups) - 1}]: ").strip()
        if raw.isdigit():
            choice = int(raw)
    return candidate_groups[choice]


def preview_final_weld_line(mesh, chain, own_faces_idx, far_faces_idx):
    pl = pv.Plotter(window_size=[1000, 800])
    pl.add_mesh(mesh, color="lightgray", opacity=0.4)
    own_sub = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], own_faces_idx))
    far_sub = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], far_faces_idx))
    pl.add_mesh(own_sub, color="red", opacity=0.7, label=f"Own-side face(s) {own_faces_idx}")
    pl.add_mesh(far_sub, color="blue", opacity=0.7, label=f"Far-side face(s) {far_faces_idx}")
    poly = _sample_chain_polyline(chain)
    pl.add_lines(poly, color="black", width=6, connected=True, label="Weld line")
    pl.add_legend()
    pl.add_title("Resolved weld line -- close the window to continue", font_size=11)
    pl.show()


def resolve_weld_line_faces(mesh, faces, face_groups, component_of_face, chain, owner_comp, other_comp, tol):
    own_idx, own_faces = resolve_own_face(chain, faces, face_groups, component_of_face, owner_comp)
    print(f"[Stage 3] Own-side (component {owner_comp}) face auto-resolved: faces {own_idx}")

    far_candidates = find_far_side_faces(chain, faces, component_of_face, face_groups, other_comp, tol)
    print(f"[Stage 3] Far-side (component {other_comp}) candidate face group(s): {far_candidates}")

    if len(far_candidates) == 1:
        far_idx = far_candidates[0]
    elif len(far_candidates) > 1:
        far_idx = pick_far_face_by_index(mesh, far_candidates)
    else:
        print("[Stage 3] No far-side face found within tolerance -- pick both faces manually.")
        print("Click the two faces adjacent to this weld line, then close the window.")
        _, _, own_idx, far_idx = pick_two_faces(mesh, faces, face_groups, base_scalars="ComponentID")
        own_faces = [faces[i] for i in own_idx]

    far_faces = [faces[i] for i in far_idx]
    preview_final_weld_line(mesh, chain, own_idx, far_idx)
    return own_idx, own_faces, far_idx, far_faces



def find_weld_lines_between(mesh, faces, face_groups, group_loops, component_of_face, comp_trees,
                             comp_a, comp_b, tol):
    print(f"\n[Stage 2] Scanning edges of component {comp_a} and component {comp_b} for weld-line "
          f"candidates (an edge/loop of one component whose points all stay within {tol} mm "
          f"of the other)...")
    candidates = scan_weld_line_candidates(comp_a, comp_b, component_of_face, faces, face_groups,
                                            group_loops, comp_trees, tol)
    print(f"[Stage 2] Found {len(candidates)} candidate edge(s)/loop(s).")
    if not candidates:
        return []

    chosen_idx = preview_and_select_candidates(
        mesh, candidates, title=f"[Stage 2] Weld-line candidates between component {comp_a} and {comp_b}")

    resolved = []
    for k in chosen_idx:
        cand = candidates[k]
        chain, owner_comp, other_comp = cand["chain"], cand["owner_comp"], cand["other_comp"]
        own_idx, own_faces, far_idx, far_faces = resolve_weld_line_faces(
            mesh, faces, face_groups, component_of_face, chain, owner_comp, other_comp, tol)

        raw_confirm = input("Keep this weld line? [Y/n]: ").strip().lower()
        if raw_confirm in ("n", "no"):
            print("Discarded.")
            continue

        resolved.append({
            "chain": chain,
            "face1": own_faces, "face2": far_faces,
            "face1_idx": own_idx, "face2_idx": far_idx,
            "comp_a": owner_comp, "comp_b": other_comp,
        })
        print(f"Weld line kept: component {owner_comp} <-> component {other_comp} "
              f"({len(chain)} edge piece(s))")

    return resolved



def main():
    parser = argparse.ArgumentParser(description="Robust weld-line identification for STEP assemblies")
    parser.add_argument("--step", required=True, help="Path to input STEP assembly file")
    parser.add_argument("--proximity_tol", type=float, default=2.0,
                         help="Max distance (mm) for an edge/loop to be considered near the other component")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    args = parser.parse_args()

    print("[Stage 0] Loading STEP assembly...")
    shape = load_step(args.step)
    solids, faces, component_of_face = identify_components(shape)
    print(f"[Stage 0] Identified {len(solids)} component(s), {len(faces)} face(s) total.")
    if len(solids) < 2:
        raise RuntimeError("This STEP file has only one solid component -- weld-line identification "
                            "needs an assembly of at least two separate components.")

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)
    add_component_ids(tagged_mesh, component_of_face)
    visualize_components(tagged_mesh)

    print("\n[Stage 1] Scanning for split-cylinder face groups...")
    face_groups = find_cylinder_face_groups(faces)
    report_cylinder_groups(face_groups)

    print("[Stage 1] Repairing edge loops for every split-cylinder group...")
    group_loops = repair_cylinder_group_loops(faces, face_groups)
    report_repaired_loops(group_loops)
    visualize_repaired_loops(tagged_mesh, group_loops)

    comp_trees = build_component_point_trees(tagged_mesh, len(solids))

    all_weld_lines = []
    while True:
        raw = input("\nFind weld lines between two components? [Y/n to finish]: ").strip().lower()
        if raw in ("n", "no"):
            break

        comp_a = confirm_or_retry(
            lambda: pick_component(tagged_mesh, component_of_face,
                                    "Left-click any face of the FIRST component, then close the window.",
                                    highlight_color="gold"),
            describe=lambda c: f"component #{c}" if c is not None else "nothing picked")
        comp_b = confirm_or_retry(
            lambda: pick_component(tagged_mesh, component_of_face,
                                    "Left-click any face of the SECOND component, then close the window.",
                                    highlight_color="cyan"),
            describe=lambda c: f"component #{c}" if c is not None else "nothing picked")
        if comp_a is None or comp_b is None or comp_a == comp_b:
            print("Need two distinct components -- try again.")
            continue

        found = find_weld_lines_between(tagged_mesh, faces, face_groups, group_loops, component_of_face,
                                         comp_trees, comp_a, comp_b, args.proximity_tol)
        all_weld_lines.extend(found)

    print(f"\nTotal weld lines identified: {len(all_weld_lines)}")
    for i, wl in enumerate(all_weld_lines, 1):
        print(f"  {i}. component {wl['comp_a']} <-> component {wl['comp_b']}, "
              f"{len(wl['chain'])} edge piece(s), faces {wl['face1_idx']} <-> {wl['face2_idx']}")


if __name__ == '__main__':
    main()
