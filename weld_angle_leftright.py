import argparse

from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    find_cylinder_face_groups,
    report_cylinder_groups,
    pick_two_faces,
    resolve_weld_seam,
    analyze_and_visualize_with_tolerance,
    prompt_for_clearance_rings,
    component_id_per_face,
    tag_component_ids,
)
from weldfinal_assembly import pick_edge_manually


def main():
    parser = argparse.ArgumentParser(
        description="Weld joint accessibility with +/- travel-angle (left/right) tolerance rays")
    parser.add_argument("--step", required=True, help="Path to input STEP file (part or assembly)")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of sample points along the weld edge")
    parser.add_argument("--near_tol", type=float, default=0.5, help="Near-field start offset in mm")
    parser.add_argument("--tol_deg", type=float, default=20.0,
                         help="Travel-angle tolerance swept left/right from the bisector, in degrees")
    parser.add_argument("--angle_step", type=float, default=None,
                         help="If set, fill the +/-tol_deg range with a ray every this many degrees "
                              "(e.g. tol_deg=20, angle_step=2 tests 21 rays/point) instead of just "
                              "the bisector and the two +/-tol_deg extremes")
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

    print("Loading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    print(f"Loaded part with {len(faces)} faces.")

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

    analyze_and_visualize_with_tolerance(
        tagged_mesh, weld_seam, face1, face2, face1_idx, face2_idx,
        num_samples=args.num_samples, near_tol=args.near_tol,
        tol_deg=args.tol_deg, mode="travelangle", angle_step=args.angle_step,
        n_rings=n_rings, clearance_radius=clearance_radius, ring_offset=ring_offset)


if __name__ == '__main__':
    main()
