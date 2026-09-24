import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    _sample_edge_polyline,
    face_surface_type_str,
)
from weld_line_finder import (
    identify_components,
    collect_candidate_edges_assembly,
    build_edge_line_mesh,
    show_labeled_components,
)

THIS_DIR = Path(__file__).resolve().parent


def prompt_component(n_components):
    """Returns a component ID, 'show' to redisplay the numbered overview, or None to quit."""
    while True:
        raw = input(f"\nComponent ID to view isolated (0-{n_components - 1}), 'show' to redisplay the "
                    f"numbered overview, or blank to quit: ").strip().lower()
        if raw == "":
            return None
        if raw in ("show", "s"):
            return "show"
        if raw.isdigit() and int(raw) < n_components:
            return int(raw)
        print(f"Enter a number between 0 and {n_components - 1}.")


def candidate_index(edge, candidate_edges):
    """The candidate-edge ID (the number weld_line_finder.py / the docx use) of this edge, or None
    if the edge isn't a weld-line candidate at all (e.g. the internal seam of a split cylinder)."""
    for i, ce in enumerate(candidate_edges):
        if ce.IsSame(edge):
            return i
    return None


def face_adjacent_edges(face_group, faces):
    edges = []
    for face_idx in face_group:
        for e in TopologyExplorer(faces[face_idx]).edges():
            if not any(e.IsSame(x) for x in edges):
                edges.append(e)
    return edges


def show_face_edges(pl, tagged_mesh, faces, face_groups, candidate_edges, face_id):
    """Highlights the picked face (its whole cylinder group, if it is one of several STEP faces
    forming one surface) and labels ONLY that face's own edges. Everything is added under a fixed
    name, so the next pick replaces the previous face's highlight/labels instead of piling up."""
    group = face_groups.get(face_id, [face_id])
    edges = face_adjacent_edges(group, faces)
    ids = [candidate_index(e, candidate_edges) for e in edges]

    face_mesh = tagged_mesh.extract_cells(np.isin(tagged_mesh.cell_data["FaceID"], group))
    pl.add_mesh(face_mesh, color="salmon", opacity=0.9, name="picked_face", pickable=False)

    candidates = [e for e, i in zip(edges, ids) if i is not None]
    others = [e for e, i in zip(edges, ids) if i is None]
    for name, edge_list, color in (("edges_candidate", candidates, "red"), ("edges_other", others, "dimgray")):
        if edge_list:
            pl.add_mesh(build_edge_line_mesh(edge_list), color=color, line_width=5,
                        render_lines_as_tubes=True, name=name, pickable=False)
        else:
            pl.remove_actor(name)

    label_points = [_sample_edge_polyline(e, 20)[10] for e in edges]
    labels = [str(i) if i is not None else "n/a" for i in ids]
    pl.add_point_labels(label_points, labels, font_size=16, text_color="black", shape_color="white",
                        shape_opacity=0.85, always_visible=True, name="edge_labels", pickable=False)

    group_note = f" (group of faces {sorted(group)})" if len(group) > 1 else ""
    surface = face_surface_type_str(faces[group[0]])
    candidate_ids = [i for i in ids if i is not None]
    summary = f"Face {face_id} [{surface}]{group_note}: edge ID(s) {candidate_ids}"
    if others:
        summary += f" + {len(others)} edge(s) with no candidate ID (n/a)"
    print(summary)
    pl.add_text(summary + "\nLeft-click another face, or close the window.", position="upper_left",
                font_size=11, name="status")


def inspect_component(tagged_mesh, faces, face_groups, candidate_edges, component_id):
    comp_mesh = tagged_mesh.extract_cells(tagged_mesh.cell_data["ComponentID"] == component_id)
    if comp_mesh.n_cells == 0:
        print(f"Component {component_id} has no mesh cells.")
        return

    pl = pv.Plotter(window_size=[1200, 900])
    pl.add_mesh(comp_mesh, color="lightgray", opacity=0.6)
    pl.add_text(f"Component {component_id} (isolated)\nLeft-click a face to label its adjacent edges "
                f"(candidate edge IDs). Close the window when done.",
                position="upper_left", font_size=11, name="status")

    def on_pick(picked):
        if picked is None or picked.n_cells == 0 or "FaceID" not in picked.cell_data:
            return
        show_face_edges(pl, tagged_mesh, faces, face_groups, candidate_edges,
                        int(picked.cell_data["FaceID"][0]))

    pl.enable_element_picking(callback=on_pick, mode="cell", left_clicking=True, show_message=False)
    pl.show()


def main():
    parser = argparse.ArgumentParser(
        description="Debug viewer: shows an assembly STEP file with its components numbered, lets you "
                    "pick one to view isolated, and there lets you click a face to show ONLY that "
                    "face's adjacent edges, labelled with their candidate-edge IDs (the same numbers "
                    "weld_line_finder.py shows and the weld-line docx refers to).")
    parser.add_argument("--step", default=str(THIS_DIR / "Assem2.STEP"), help="Path to input STEP file")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    args = parser.parse_args()

    print("Loading STEP file...")
    shape = load_step(args.step)
    solids, faces, component_of_face = identify_components(shape)
    n_components = len(solids)
    print(f"{n_components} component(s), {len(faces)} face(s) total.")

    face_groups = find_cylinder_face_groups(faces, component_of_face=component_of_face)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)
    lookup = np.array(component_of_face, dtype=np.int32)
    tagged_mesh.cell_data["ComponentID"] = lookup[tagged_mesh.cell_data["FaceID"]]

    candidate_edges, _own_info = collect_candidate_edges_assembly(shape, faces, face_groups, component_of_face)

    show_labeled_components(tagged_mesh, n_components)
    while True:
        choice = prompt_component(n_components)
        if choice is None:
            break
        if choice == "show":
            show_labeled_components(tagged_mesh, n_components)
            continue
        inspect_component(tagged_mesh, faces, face_groups, candidate_edges, choice)


if __name__ == "__main__":
    main()
