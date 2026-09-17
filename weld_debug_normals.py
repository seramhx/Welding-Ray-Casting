import argparse

import numpy as np
import pyvista as pv

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.TopologyUtils import TopologyExplorer

from weldfinal import (
    load_step,
    build_face_tagged_mesh,
    build_face_normal_mesh,
    build_per_face_normal_meshes,
    find_cylinder_face_groups,
    report_cylinder_groups,
    compute_group_normal_signs_by_raycast,
    apply_normal_sign_correction,
    _id_to_face_index,
    pick_two_faces,
    resolve_weld_seam,
    bbox_diagonal,
    _sample_chain_polyline,
    _edge_length_estimate,
    _pcurve_uv,
    _surface_point_and_normal,
    _mesh_facet_point_and_normal,
    face_surface_type_str,
)
from weldfinal_assembly import pick_edge_manually



def compute_debug_data(chain, face1, face2, mesh_fallback1=None, mesh_fallback2=None, num_samples=15):
    lengths = [_edge_length_estimate(edge) for edge, _ in chain]
    total_length = sum(lengths) or 1.0
    counts = [max(2, round(num_samples * l / total_length)) for l in lengths]

    edge_pts, proj1_pts, proj2_pts, n1_vecs, n2_vecs, bisectors = [], [], [], [], [], []
    uv_source1, uv_source2, normal_defined1, normal_defined2 = [], [], [], []
    owner1_ids, owner2_ids = [], []

    for piece_idx, ((edge, reversed_), n_i) in enumerate(zip(chain, counts)):
        adaptor = BRepAdaptor_Curve(edge)
        u_min, u_max = adaptor.FirstParameter(), adaptor.LastParameter()
        params = np.linspace(u_min, u_max, n_i)
        if reversed_:
            params = params[::-1]
        start = 0 if piece_idx == 0 else 1

        for p in params[start:]:
            gp_p, gp_v = gp_Pnt(), gp_Vec()
            adaptor.D1(p, gp_p, gp_v)
            p3d = np.array([gp_p.X(), gp_p.Y(), gp_p.Z()], dtype=np.float32)

            uv1, owner1 = _pcurve_uv(edge, face1, p)
            uv2, owner2 = _pcurve_uv(edge, face2, p)

            if uv1 is not None:
                proj1, n1, nd1 = _surface_point_and_normal(owner1, uv1, gp_p)
                src1, oid1 = "pcurve", id(owner1)
            elif mesh_fallback1 is not None:
                proj1, n1 = _mesh_facet_point_and_normal(mesh_fallback1, gp_p)
                nd1, src1, oid1 = True, "mesh", None
            else:
                fallback_face1 = face1[0] if isinstance(face1, (list, tuple)) else face1
                proj1, n1, nd1 = _surface_point_and_normal(fallback_face1, None, gp_p)
                src1, oid1 = "projected", None

            if uv2 is not None:
                proj2, n2, nd2 = _surface_point_and_normal(owner2, uv2, gp_p)
                src2, oid2 = "pcurve", id(owner2)
            elif mesh_fallback2 is not None:
                proj2, n2 = _mesh_facet_point_and_normal(mesh_fallback2, gp_p)
                nd2, src2, oid2 = True, "mesh", None
            else:
                fallback_face2 = face2[0] if isinstance(face2, (list, tuple)) else face2
                proj2, n2, nd2 = _surface_point_and_normal(fallback_face2, None, gp_p)
                src2, oid2 = "projected", None

            bis = n1 + n2
            b_norm = np.linalg.norm(bis)
            bis = bis / b_norm if b_norm > 1e-6 else n1

            edge_pts.append(p3d)
            proj1_pts.append(proj1)
            proj2_pts.append(proj2)
            n1_vecs.append(n1)
            n2_vecs.append(n2)
            bisectors.append(bis)
            uv_source1.append(src1)
            uv_source2.append(src2)
            normal_defined1.append(nd1)
            normal_defined2.append(nd2)
            owner1_ids.append(oid1)
            owner2_ids.append(oid2)

    return {
        "edge_pts": np.array(edge_pts, dtype=np.float32),
        "proj1_pts": np.array(proj1_pts, dtype=np.float32),
        "proj2_pts": np.array(proj2_pts, dtype=np.float32),
        "n1": np.array(n1_vecs, dtype=np.float32),
        "n2": np.array(n2_vecs, dtype=np.float32),
        "bisector": np.array(bisectors, dtype=np.float32),
        "uv_source1": uv_source1,
        "uv_source2": uv_source2,
        "normal_defined1": normal_defined1,
        "normal_defined2": normal_defined2,
        "owner1_ids": owner1_ids,
        "owner2_ids": owner2_ids,
    }


def apply_raycast_sign_correction(mesh_fallback1, mesh_fallback2, data, id_to_idx=None,
                                   per_face_meshes1=None, per_face_meshes2=None):
    id_to_idx = id_to_idx or {}
    signs1 = compute_group_normal_signs_by_raycast(mesh_fallback1, data["edge_pts"], data["n1"], data["owner1_ids"],
                                                     per_face_meshes=per_face_meshes1)
    signs2 = compute_group_normal_signs_by_raycast(mesh_fallback2, data["edge_pts"], data["n2"], data["owner2_ids"],
                                                     per_face_meshes=per_face_meshes2)
    flipped = sorted({id_to_idx.get(oid, oid) for s in (signs1, signs2) for oid, v in s.items() if v < 0})

    corrected = dict(data)
    corrected["n1"] = apply_normal_sign_correction(data["n1"], data["owner1_ids"], signs1)
    corrected["n2"] = apply_normal_sign_correction(data["n2"], data["owner2_ids"], signs2)
    bis = corrected["n1"] + corrected["n2"]
    bis_norms = np.linalg.norm(bis, axis=1, keepdims=True)
    bis_norms[bis_norms < 1e-6] = 1.0
    corrected["bisector"] = bis / bis_norms
    return corrected, flipped


def compute_mesh_facet_normals(face1_pv, face2_pv, data):
    mesh_n1, mesh_n2, dot1, dot2 = [], [], [], []
    for i in range(len(data["edge_pts"])):
        c1 = face1_pv.find_closest_cell(data["proj1_pts"][i])
        c2 = face2_pv.find_closest_cell(data["proj2_pts"][i])
        n1m = np.array(face1_pv.cell_data["Normals"][c1], dtype=np.float32)
        n2m = np.array(face2_pv.cell_data["Normals"][c2], dtype=np.float32)
        mesh_n1.append(n1m)
        mesh_n2.append(n2m)
        dot1.append(float(np.dot(n1m, data["n1"][i])))
        dot2.append(float(np.dot(n2m, data["n2"][i])))

    return {
        "mesh_n1": np.array(mesh_n1, dtype=np.float32),
        "mesh_n2": np.array(mesh_n2, dtype=np.float32),
        "dot1": np.array(dot1, dtype=np.float32),
        "dot2": np.array(dot2, dtype=np.float32),
    }


def _face_group_label(face_or_faces):
    faces = face_or_faces if isinstance(face_or_faces, (list, tuple)) else [face_or_faces]
    orient = "REVERSED" if faces[0].Orientation() == TopAbs_REVERSED else "FORWARD"
    label = f"surface type = {face_surface_type_str(faces[0])}, TopAbs orientation = {orient}"
    if len(faces) > 1:
        label += f" (grouped, {len(faces)} STEP faces on the same cylinder)"
    return label


def print_debug_table(data, face1, face2, mesh_check=None):
    edge_pts, proj1, proj2, n1, n2 = data["edge_pts"], data["proj1_pts"], data["proj2_pts"], data["n1"], data["n2"]
    drift1 = np.linalg.norm(edge_pts - proj1, axis=1)
    drift2 = np.linalg.norm(edge_pts - proj2, axis=1)
    dot_n = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_n))

    print(f"\nFace 1: {_face_group_label(face1)}")
    print(f"Face 2: {_face_group_label(face2)}")

    has_check = mesh_check is not None
    header = f"{'#':>3} | {'face1 UV':>10} | {'drift1':>9} | {'face2 UV':>10} | {'drift2':>9} | {'n1/n2 angle':>12}"
    if has_check:
        header += f" | {'mesh.n1 dot':>12} | {'mesh.n2 dot':>12}"
    width = len(header)

    print("\n" + "=" * width)
    print(header)
    print("-" * width)
    for i in range(len(edge_pts)):
        row = (f"{i:>3} | {data['uv_source1'][i]:>10} | {drift1[i]:>9.4f} | "
               f"{data['uv_source2'][i]:>10} | {drift2[i]:>9.4f} | {angle_deg[i]:>12.2f}")
        if has_check:
            row += f" | {mesh_check['dot1'][i]:>12.3f} | {mesh_check['dot2'][i]:>12.3f}"
        if not data["normal_defined1"][i] or not data["normal_defined2"][i]:
            row += "   <-- NORMAL UNDEFINED (degenerate point, fell back to [0,0,1])"
        print(row)
    print("=" * width)
    print(f"Max projection drift: face1={drift1.max():.4f}  face2={drift2.max():.4f}")
    print("'pcurve'    = exact UV from the edge's surface representation (reliable).")
    print("'mesh'      = nearest-mesh-facet normal (gmsh triangle winding), used when the edge has")
    print("              no representation on that face -- robust even on periodic surfaces.")
    print("'projected' = analytic nearest-point search (ShapeAnalysis_Surface.ValueOfUV), only used")
    print("              when no mesh fallback was supplied -- can be unreliable on periodic seams")
    print("              (cylinders, etc), converging to the wrong angular position.")
    if has_check:
        print("'mesh.n dot' = analytic normal . raw mesh-triangle-winding normal at the same facet,")
        print("              independent of B-Rep orientation bookkeeping. ~+1 = consistent,")
        print("              ~-1 = the analytic normal is inverted relative to the mesh, ~0 = the")
        print("              nearest facet found is probably not the right one.")
        if np.any(mesh_check["dot1"] < -0.3):
            print("  >>> Face 1 analytic normal disagrees with mesh winding at some points (possible inversion).")
        if np.any(mesh_check["dot2"] < -0.3):
            print("  >>> Face 2 analytic normal disagrees with mesh winding at some points (possible inversion).")
    print()



def render_debug(mesh, face1_idx, face2_idx, weld_seam, data, mesh_check=None, arrow_mag=None):
    idx1 = face1_idx if isinstance(face1_idx, (list, tuple)) else [face1_idx]
    idx2 = face2_idx if isinstance(face2_idx, (list, tuple)) else [face2_idx]
    face1_pv = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], idx1))
    face2_pv = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], idx2))

    if arrow_mag is None:
        arrow_mag = 0.04 * bbox_diagonal(mesh)

    pl = pv.Plotter(window_size=[1200, 900])
    pl.add_mesh(mesh, color="lightgray", opacity=0.3)
    pl.add_mesh(face1_pv, color="red", opacity=0.45, label=f"Face(s) {idx1}")
    pl.add_mesh(face2_pv, color="blue", opacity=0.45, label=f"Face(s) {idx2}")

    edge_line = _sample_chain_polyline(weld_seam)
    pl.add_lines(edge_line, color="black", width=5, connected=True, label="Weld seam")
    pl.add_points(data["edge_pts"], color="yellow", point_size=12, render_points_as_spheres=True,
                  label="Edge sample points")

    facet_added_to_legend = False
    mesh_normal_added_to_legend = False
    for i in range(len(data["edge_pts"])):
        ep = data["edge_pts"][i]
        p1, p2 = data["proj1_pts"][i], data["proj2_pts"][i]
        n1, n2, bis = data["n1"][i], data["n2"][i], data["bisector"][i]

        pl.add_lines(np.array([ep, p1], dtype=np.float32), color="cyan", width=1)
        pl.add_lines(np.array([ep, p2], dtype=np.float32), color="orange", width=1)

        pl.add_points(np.array([p1]), color="cyan", point_size=8, render_points_as_spheres=True)
        pl.add_points(np.array([p2]), color="orange", point_size=8, render_points_as_spheres=True)

        pl.add_arrows(p1, n1, mag=arrow_mag, color="cyan")
        pl.add_arrows(p2, n2, mag=arrow_mag, color="orange")
        pl.add_arrows(ep, bis, mag=arrow_mag * 1.5, color="red")

        cell1 = face1_pv.extract_cells([face1_pv.find_closest_cell(p1)])
        cell2 = face2_pv.extract_cells([face2_pv.find_closest_cell(p2)])
        pl.add_mesh(cell1, color="magenta", style="wireframe", line_width=4,
                    label="Tested facet" if not facet_added_to_legend else None)
        pl.add_mesh(cell2, color="magenta", style="wireframe", line_width=4)
        facet_added_to_legend = True

        if mesh_check is not None:
            pl.add_arrows(p1, mesh_check["mesh_n1"][i], mag=arrow_mag * 0.8, color="lime",
                          label="Mesh facet normal" if not mesh_normal_added_to_legend else None)
            pl.add_arrows(p2, mesh_check["mesh_n2"][i], mag=arrow_mag * 0.8, color="lime")
            mesh_normal_added_to_legend = True

    pl.add_legend()
    title = (
        "Weld Normal/Bisector Debug\n"
        "Cyan = face1 normal, Orange = face2 normal, Red = bisector, Magenta wireframe = tested facet"
    )
    if mesh_check is not None:
        title += "\nLime = raw mesh-triangle normal at that facet (compare against cyan/orange for inversion)"
    pl.add_title(title, font_size=10)
    pl.show()



def main():
    parser = argparse.ArgumentParser(description="Debug visualization for weld normals and bisector")
    parser.add_argument("--step", required=True, help="Path to input STEP file")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of sample points along the weld edge")
    parser.add_argument("--minh", type=float, default=0.5, help="Gmsh min element size (mm)")
    parser.add_argument("--maxh", type=float, default=3.0, help="Gmsh max element size (mm)")
    parser.add_argument("--curvature", type=float, default=10, help="Gmsh curvature-based mesh refinement factor")
    parser.add_argument("--force_remesh", action="store_true", help="Ignore any cached mesh and re-mesh via Gmsh")
    parser.add_argument("--arrow_scale", type=float, default=0.04,
                         help="Normal/bisector arrow length as a fraction of the part bbox diagonal")
    args = parser.parse_args()

    print("Loading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    print(f"Loaded part with {len(faces)} faces.")

    face_groups = find_cylinder_face_groups(faces)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)

    print("\nOpening interactive window: left-click two adjacent faces to inspect the weld joint.")
    face1, face2, face1_idx, face2_idx = pick_two_faces(tagged_mesh, faces, face_groups)
    print(f"Selected faces: {face1_idx} and {face2_idx}")

    weld_seam = resolve_weld_seam(topo, tagged_mesh, face1, face2, allow_manual_pick=True)
    if weld_seam is None:
        print("The two selected faces share no B-Rep topology (separate assembly components).")
        weld_seam = pick_edge_manually(tagged_mesh, topo, face1, face2)

    mesh_fallback1 = build_face_normal_mesh(tagged_mesh, face1_idx)
    mesh_fallback2 = build_face_normal_mesh(tagged_mesh, face2_idx)
    per_face_meshes1 = build_per_face_normal_meshes(tagged_mesh, face1, face1_idx)
    per_face_meshes2 = build_per_face_normal_meshes(tagged_mesh, face2, face2_idx)

    data = compute_debug_data(
        weld_seam, face1, face2,
        mesh_fallback1=mesh_fallback1, mesh_fallback2=mesh_fallback2, num_samples=args.num_samples)

    id_to_idx = {}
    id_to_idx.update(_id_to_face_index(face1, face1_idx))
    id_to_idx.update(_id_to_face_index(face2, face2_idx))
    data, flipped = apply_raycast_sign_correction(mesh_fallback1, mesh_fallback2, data, id_to_idx,
                                                   per_face_meshes1=per_face_meshes1,
                                                   per_face_meshes2=per_face_meshes2)
    if flipped:
        print(f"Corrected inward-pointing normal(s) via mesh-winding cross-check (majority vote "
              f"over the seam's own sample points): faces {flipped}")

    mesh_check = compute_mesh_facet_normals(mesh_fallback1, mesh_fallback2, data)
    print_debug_table(data, face1, face2, mesh_check=mesh_check)

    arrow_mag = args.arrow_scale * bbox_diagonal(tagged_mesh)
    render_debug(tagged_mesh, face1_idx, face2_idx, weld_seam, data, mesh_check=mesh_check, arrow_mag=arrow_mag)


if __name__ == '__main__':
    main()
