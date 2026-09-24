
import argparse

from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    pick_two_faces,
    pick_edge_by_index,
    resolve_weld_seam,
    analyze_and_visualize,
    prompt_for_clearance_rings,
    component_id_per_face,
    tag_component_ids,
)


def pick_edge_manually(mesh, topo, face1, face2):
    faces1 = face1 if isinstance(face1, (list, tuple)) else [face1]
    faces2 = face2 if isinstance(face2, (list, tuple)) else [face2]
    edges1 = [e for f in faces1 for e in topo.edges_from_face(f)]
    edges2 = [e for f in faces2 for e in topo.edges_from_face(f)]
    candidates = [[(e, False)] for e in (edges1 + edges2)]
    edge_colors = ["cyan"] * len(edges1) + ["orange"] * len(edges2)

    title = (
        "No shared edge between the two picked faces (separate assembly components).\n"
        "Cyan = boundary edges of face 1, Orange = boundary edges of face 2.\n"
        "Note the index of the weld edge, then close the window."
    )
    return pick_edge_by_index(mesh, candidates, title, chain_colors=edge_colors)


def main():
    parser = argparse.ArgumentParser(
        description="Weld Joint Accessibility Pipeline for STEP assemblies with non-touching components")
    parser.add_argument("--step", required=True, help="Path to input STEP assembly file")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of sample points along the weld edge")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Near-field start offset in mm")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    parser.add_argument("--n_rings", type=int, default=None,
                         help="Clearance ring rays per point simulating a spherical tool radius "
                              "(skips the interactive prompt below; 0 disables)")
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

    print("Loading STEP assembly...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    print(f"Loaded assembly with {len(faces)} faces.")

    face_groups = find_cylinder_face_groups(faces)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)

    component_of_face, n_components = component_id_per_face(shape, faces)
    if n_components > 1:
        tag_component_ids(tagged_mesh, component_of_face)
        print(f"{n_components} component(s) detected -- shown in different colors.")

    print("\nOpening interactive window: left-click two adjacent faces (from either component) to select the weld joint.")
    face1, face2, face1_idx, face2_idx = pick_two_faces(tagged_mesh, faces, face_groups)
    print(f"Selected faces: {face1_idx} and {face2_idx}")

    weld_seam = resolve_weld_seam(topo, tagged_mesh, face1, face2, allow_manual_pick=True)
    if weld_seam is None:
        print("The two selected faces share no B-Rep topology (separate assembly components).")
        weld_seam = pick_edge_manually(tagged_mesh, topo, face1, face2)

    analyze_and_visualize(tagged_mesh, weld_seam, face1, face2, face1_idx, face2_idx,
                          num_samples=args.num_samples, near_tol=args.near_tol,
                          n_rings=n_rings, clearance_radius=clearance_radius,
                          ring_offset=ring_offset)


if __name__ == '__main__':
    main()
