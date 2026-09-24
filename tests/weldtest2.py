import os
import sys
import random
import argparse
import tempfile
import numpy as np
import pyvista as pv
import warp as wp
import gmsh

# ── OpenCASCADE Imports ──────────────────────────────────────────────────────
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
from OCC.Core import TopoDS
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepOffset import BRepOffset_Analyse
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Extend.TopologyUtils import TopologyExplorer


# ─────────────────────────────────────────────────────────────────────────────
# 1. GMSH HIGH-QUALITY MESH GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def mesh_step_with_gmsh(step_path, minh=0.2, maxh=2.0, curvature=12):
    """Mesh a STEP model using Gmsh to generate a high-density triangular mesh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)  # Quiet output
    gmsh.model.add("weld_model")
    
    # Import STEP into Gmsh OCC kernel
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    
    # Fine mesh settings
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
    gmsh.option.setNumber("Mesh.MeshSizeMin", minh)
    gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D", True)
    
    # Extract nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(coords, dtype=np.float32).reshape(-1, 3)
    tag_map = {tag: idx for idx, tag in enumerate(node_tags)}
    
    # Extract 2D triangular elements
    elem_types, elem_tags, node_tags_list = gmsh.model.mesh.getElements(dim=2)
    triangles = []
    for etype, etags, ntags in zip(elem_types, elem_tags, node_tags_list):
        if gmsh.model.mesh.getElementProperties(etype)[3] == 3:  # Triangle
            tris = np.array(ntags, dtype=int).reshape(-1, 3)
            for t in tris:
                triangles.append([3, tag_map[t[0]], tag_map[t[1]], tag_map[t[2]]])
                
    gmsh.finalize()
    
    tris_np = np.hstack(triangles).astype(np.int32)
    return pv.PolyData(nodes, tris_np)


def mesh_occ_shape_with_gmsh(occ_shape, minh=0.2, maxh=2.0, curvature=12):
    """Mesh an individual OCC shape (e.g., single face) using Gmsh."""
    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as tmp:
        tmp_path = tmp.name
    
    breptools.Write(occ_shape, tmp_path)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("sub_shape")
    gmsh.model.occ.importShapes(tmp_path)
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
    gmsh.option.setNumber("Mesh.MeshSizeMin", minh)
    gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
    gmsh.model.mesh.generate(2)
    
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(coords, dtype=np.float32).reshape(-1, 3)
    tag_map = {tag: idx for idx, tag in enumerate(node_tags)}
    
    elem_types, elem_tags, node_tags_list = gmsh.model.mesh.getElements(dim=2)
    triangles = []
    for etype, etags, ntags in zip(elem_types, elem_tags, node_tags_list):
        if gmsh.model.mesh.getElementProperties(etype)[3] == 3:
            tris = np.array(ntags, dtype=int).reshape(-1, 3)
            for t in tris:
                triangles.append([3, tag_map[t[0]], tag_map[t[1]], tag_map[t[2]]])
                
    gmsh.finalize()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    if len(nodes) == 0 or len(triangles) == 0:
        return None
        
    tris_np = np.hstack(triangles).astype(np.int32)
    return pv.PolyData(nodes, tris_np)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CAD LOADING & EDGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def load_step(step_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise FileNotFoundError(f"Could not read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def extract_concave_edges(shape):
    """Find and deduplicate all concave edges (weld lines) via BRepOffset_Analyse."""
    analyser = BRepOffset_Analyse(shape, 0.01)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    
    concave_edges = []
    
    while exp.More():
        face = TopoDS.Face(exp.Current())
        c_list = TopTools_ListOfShape()
        analyser.Edges(face, 0, c_list)  # 0 = Concave
        
        it = TopTools_ListIteratorOfListOfShape(c_list)
        while it.More():
            edge = TopoDS.Edge(it.Value())
            if not any(edge.IsSame(e) for e in concave_edges):
                concave_edges.append(edge)
            it.Next()
        exp.Next()
        
    return concave_edges


# ─────────────────────────────────────────────────────────────────────────────
# 3. B-REP GEOMETRY COMPUTATIONS & LOCAL FRAME SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def get_face_normal_at_point(face, pnt):
    """Calculate outward surface normal of a face at a 3D point."""
    surf_handle = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surf_handle)
    uv = sas.ValueOfUV(pnt, 1e-3)
    u, v = uv.Coord(1), uv.Coord(2)
    
    adaptor = BRepAdaptor_Surface(face, True)
    props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 1, 1e-4)
    if props.IsNormalDefined():
        n = props.Normal()
        vec = np.array([n.X(), n.Y(), n.Z()], dtype=np.float32)
        if face.Orientation() == TopAbs_REVERSED:
            vec = -vec
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def sample_edge_and_get_bisectors(edge, shape, num_samples=15):
    """Sample points along a concave edge and calculate face normals, bisectors, and tangents."""
    adaptor = BRepAdaptor_Curve(edge)
    u_min, u_max = adaptor.FirstParameter(), adaptor.LastParameter()
    params = np.linspace(u_min, u_max, num_samples)
    
    topo = TopologyExplorer(shape)
    adj_faces = list(topo.faces_from_edge(edge))
    
    pts, normals1, normals2, bisectors, tangents = [], [], [], [], []
    
    for p in params:
        gp_p = gp_Pnt()
        gp_v = gp_Vec()
        adaptor.D1(p, gp_p, gp_v)  # Evaluate point and 1st derivative (tangent)
        
        p3d = np.array([gp_p.X(), gp_p.Y(), gp_p.Z()], dtype=np.float32)
        t_vec = np.array([gp_v.X(), gp_v.Y(), gp_v.Z()], dtype=np.float32)
        t_norm = np.linalg.norm(t_vec)
        t_vec = t_vec / t_norm if t_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        pts.append(p3d)
        tangents.append(t_vec)
        
        if len(adj_faces) >= 2:
            n1 = get_face_normal_at_point(adj_faces[0], gp_p)
            n2 = get_face_normal_at_point(adj_faces[1], gp_p)
            bis = n1 + n2
            norm = np.linalg.norm(bis)
            bis = bis / norm if norm > 1e-6 else n1
        else:
            n1 = np.array([0, 0, 1], dtype=np.float32)
            n2 = np.array([0, 0, 1], dtype=np.float32)
            bis = n1
            
        normals1.append(n1)
        normals2.append(n2)
        bisectors.append(bis)
        
    return (np.array(pts, dtype=np.float32), 
            np.array(normals1, dtype=np.float32), 
            np.array(normals2, dtype=np.float32), 
            np.array(bisectors, dtype=np.float32),
            np.array(tangents, dtype=np.float32),
            adj_faces)


# ─────────────────────────────────────────────────────────────────────────────
# 4. METHOD 2: PARAMETRIC DIRECTION SAMPLING (DIHEDRAL WEDGE)
# ─────────────────────────────────────────────────────────────────────────────

def generate_parametric_weld_directions(n1_avg, n2_avg, tangent_avg, n_dirs=120, max_push_drag_deg=60.0):
    """
    Generate candidate directions bounded strictly within the open dihedral wedge
    of the joint (between adjacent face planes) and swept along push/drag angles.
    """
    n1 = n1_avg / np.linalg.norm(n1_avg)
    n2 = n2_avg / np.linalg.norm(n2_avg)
    
    # 1. Bisector vector (u)
    u_vec = n1 + n2
    u_norm = np.linalg.norm(u_vec)
    u_vec = u_vec / u_norm if u_norm > 1e-6 else n1
    
    # 2. Transverse vector (v)
    v_vec = n1 - n2
    v_norm = np.linalg.norm(v_vec)
    if v_norm < 1e-6:
        v_vec = np.cross(tangent_avg, u_vec)
        v_norm_c = np.linalg.norm(v_vec)
        v_vec = v_vec / v_norm_c if v_norm_c > 1e-6 else np.array([0, 1, 0], dtype=np.float32)
    else:
        v_vec /= v_norm

    # 3. Seam Tangent vector (t) - Orthogonalized to u and v
    t_vec = tangent_avg - np.dot(tangent_avg, u_vec) * u_vec - np.dot(tangent_avg, v_vec) * v_vec
    t_norm = np.linalg.norm(t_vec)
    if t_norm < 1e-6:
        t_vec = np.cross(u_vec, v_vec)
        t_vec /= np.linalg.norm(t_vec)
    else:
        t_vec /= t_norm

    # Half-angle between n1 and n2
    dot_n = np.clip(np.dot(n1, n2), -1.0, 1.0)
    theta_half = 0.5 * np.arccos(dot_n)
    
    # Work angle limit (bounded strictly inside the open dihedral wedge)
    phi_max = (np.pi / 2.0) - theta_half
    
    # Push/drag angle limit
    beta_max = np.radians(max_push_drag_deg)
    
    # Grid resolution allocation
    n_work = int(np.sqrt(n_dirs * 1.2))
    n_push_drag = max(1, n_dirs // n_work)
    
    phi_vals = np.linspace(-phi_max, phi_max, n_work)
    beta_vals = np.linspace(-beta_max, beta_max, n_push_drag)
    
    directions = []
    for beta in beta_vals:
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        for phi in phi_vals:
            cos_p, sin_p = np.cos(phi), np.sin(phi)
            # Parametric direction vector combining work angle (in u-v plane) and push/drag angle (along t)
            d = cos_b * (cos_p * u_vec + sin_p * v_vec) + sin_b * t_vec
            d /= np.linalg.norm(d)
            directions.append(d)
            
    return np.array(directions, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. WARP RAY CASTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def _torch_ray_cast_kernel(
    mesh       : wp.uint64,
    origins    : wp.array(dtype=wp.vec3),
    directions : wp.array(dtype=wp.vec3),
    accessible : wp.array(dtype=int),
    hit_dists  : wp.array(dtype=float),
    max_dist   : float,
    near_tol   : float
):
    tid = wp.tid()
    # Apply near-field tolerance offset to tolerate minor initial corner clipping
    ray_origin = origins[tid] + directions[tid] * near_tol
    ray_dir = directions[tid]
    
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)
    
    if wp.mesh_query_ray(mesh, ray_origin, ray_dir, max_dist, t, u, v, sign, normal, face):
        accessible[tid] = 0  # Obstructed
        hit_dists[tid] = t + near_tol  # Total distance from true origin to hit
    else:
        accessible[tid] = 1  # Unobstructed
        hit_dists[tid] = 1e9  # Sentinel for no hit


def run_gpu_ray_casting(gmsh_pv_mesh, origins, directions, tool_radius=5.0, n_ring=8, near_tol=1.0):
    """Ray casting using fine Gmsh mesh and Warp BVH engine."""
    wp.init()
    
    faces = gmsh_pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32).flatten()
    verts_wp = wp.array(gmsh_pv_mesh.points.astype(np.float32), dtype=wp.vec3, device='cuda')
    faces_wp = wp.array(faces, dtype=int, device='cuda')
    wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp)
    
    bbox = gmsh_pv_mesh.bounds
    diag = np.linalg.norm([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
    max_dist = float(diag * 2.0)
    
    n_pts, n_dirs = len(origins), len(directions)
    ring_angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
    
    all_origins, all_dirs = [], []
    
    for p in origins:
        for d in directions:
            # Main center ray
            all_origins.append(p)
            all_dirs.append(d)
            
            # Ring clearance rays for torch nozzle radius
            if abs(d[0]) < 0.9:
                p1 = np.cross(d, [1, 0, 0])
            else:
                p1 = np.cross(d, [0, 1, 0])
            p1 /= np.linalg.norm(p1)
            p2 = np.cross(d, p1)
            p2 /= np.linalg.norm(p2)
            
            for a in ring_angles:
                offset = (p1 * np.cos(a) + p2 * np.sin(a)) * tool_radius
                lift = d * tool_radius * 0.5
                all_origins.append(p + offset + lift)
                all_dirs.append(d)
                
    orig_np = np.array(all_origins, dtype=np.float32)
    dirs_np = np.array(all_dirs, dtype=np.float32)
    
    orig_wp = wp.array(orig_np, dtype=wp.vec3, device='cuda')
    dirs_wp = wp.array(dirs_np, dtype=wp.vec3, device='cuda')
    acc_wp = wp.zeros(len(orig_np), dtype=int, device='cuda')
    hit_wp = wp.zeros(len(orig_np), dtype=float, device='cuda')
    
    wp.launch(
        kernel=_torch_ray_cast_kernel,
        dim=len(orig_np),
        inputs=[wp_mesh.id, orig_wp, dirs_wp, acc_wp, hit_wp, max_dist, float(near_tol)],
        device='cuda'
    )
    
    acc_res = acc_wp.numpy().reshape(n_pts, n_dirs, 1 + n_ring)
    hit_res = hit_wp.numpy().reshape(n_pts, n_dirs, 1 + n_ring)
    
    is_accessible = acc_res.all(axis=2)  # All rays in tool bundle must clear
    min_hit_dists = hit_res.min(axis=2)  # Earliest hit distance across tool bundle
    
    return is_accessible, min_hit_dists, bbox


def compute_bbox_exit_lengths(origins, directions, bbox):
    """Compute exit ray distances from edge origins to part bounding box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    lengths = np.zeros((len(origins), len(directions)), dtype=np.float32)
    
    for i, p in enumerate(origins):
        for j, d in enumerate(directions):
            t_vals = []
            for coord, bound in zip([0, 0, 1, 1, 2, 2], [xmin, xmax, ymin, ymax, zmin, zmax]):
                if abs(d[coord]) > 1e-6:
                    t = (bound - p[coord]) / d[coord]
                    if t > 0:
                        t_vals.append(t)
            lengths[i, j] = min(t_vals) if t_vals else 0.0
            
    return lengths


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weld Line Accessibility Pipeline (Method 2 - Dihedral Wedge)")
    parser.add_argument("--step", required=True, help="Path to input STEP file")
    parser.add_argument("--edge_index", type=int, default=-1, help="Index of concave edge to test (0-based). Default: -1 (random selection)")
    parser.add_argument("--tool_radius", type=float, default=5.0, help="Torch clearance radius in mm")
    parser.add_argument("--near_tol", type=float, default=1.0, help="Near-field start tolerance in mm")
    parser.add_argument("--n_dirs", type=int, default=120, help="Target number of candidate wedge directions")
    parser.add_argument("--max_push_drag", type=float, default=60.0, help="Max push/drag angle sweep (deg)")
    args = parser.parse_args()

    # Step 1: Load B-Rep Topology and Classify Edges
    shape = load_step(args.step)
    concave_edges = extract_concave_edges(shape)
    
    n_concave = len(concave_edges)
    if n_concave == 0:
        print("No concave edges (weld lines) detected in STEP model.")
        return

    print(f"\nDetected {n_concave} concave weld line(s) in STEP model (Indices 0 to {n_concave - 1}).")

    # Step 2: Handle Edge Selection (User Choice or Random)
    if 0 <= args.edge_index < n_concave:
        selected_idx = args.edge_index
        print(f"--> Using user-specified Weld Line Index: {selected_idx}")
    else:
        if args.edge_index >= n_concave:
            print(f"--> Warning: --edge_index {args.edge_index} is out of range (0 to {n_concave - 1}). Picking randomly instead.")
        selected_idx = random.randint(0, n_concave - 1)
        print(f"--> Randomly selected Weld Line Index: {selected_idx}")

    selected_edge = concave_edges[selected_idx]

    # Step 3: Fine Mesh Generation via Gmsh
    print("Generating fine mesh via Gmsh...")
    gmsh_mesh = mesh_step_with_gmsh(args.step)

    # ── Vis 1: All Potential Weld Lines ─────────────────────────────────────
    pl1 = pv.Plotter(window_size=[1000, 800])
    pl1.add_mesh(gmsh_mesh, color='lightgray', opacity=0.7, show_edges=True, edge_color='dimgray')
    
    for idx, e in enumerate(concave_edges):
        pts, _, _, _, _, _ = sample_edge_and_get_bisectors(e, shape, num_samples=30)
        line_color = 'magenta' if idx == selected_idx else 'yellow'
        line_w = 6 if idx == selected_idx else 3
        pl1.add_lines(pts, color=line_color, width=line_w)
        
    pl1.add_title(f"1. Potential Weld Lines (Magenta = Selected Index {selected_idx})", font_size=12)
    pl1.show()

    # Step 4: Sample Selected Edge & Compute Local Frame Vectors
    pts, n1_arr, n2_arr, bis_arr, tan_arr, adj_faces = sample_edge_and_get_bisectors(
        selected_edge, shape, num_samples=12)
    
    n1_avg = n1_arr.mean(axis=0)
    n2_avg = n2_arr.mean(axis=0)
    tan_avg = tan_arr.mean(axis=0)
    bis_avg = bis_arr.mean(axis=0)

    # ── Vis 2: Selected Edge, Adjacent Surfaces & Local Frame ────────────────
    pl2 = pv.Plotter(window_size=[1000, 800])
    pl2.add_mesh(gmsh_mesh, color='whitesmoke', opacity=0.4)
    
    colors = ['skyblue', 'gold']
    for idx, face in enumerate(adj_faces[:2]):
        face_pv = mesh_occ_shape_with_gmsh(face)
        if face_pv is not None:
            pl2.add_mesh(face_pv, color=colors[idx], opacity=0.8, label=f"Adjacent Face {idx+1}")

    pl2.add_points(pts, color='blue', point_size=10, render_points_as_spheres=True)
    
    for p, n1, n2, b, t in zip(pts, n1_arr, n2_arr, bis_arr, tan_arr):
        pl2.add_arrows(p, n1, mag=10.0, color='cyan')
        pl2.add_arrows(p, n2, mag=10.0, color='orange')
        pl2.add_arrows(p, b, mag=15.0, color='red')
        pl2.add_arrows(p, t, mag=15.0, color='green')

    pl2.add_legend()
    pl2.add_title(f"2. Selected Weld Line {selected_idx} (Red = Bisector, Green = Tangent, Cyan/Orange = Normals)", font_size=12)
    pl2.show()

    # Step 5: Method 2 Direction Generation & Ray Casting
    directions = generate_parametric_weld_directions(
        n1_avg, n2_avg, tan_avg, n_dirs=args.n_dirs, max_push_drag_deg=args.max_push_drag)
    
    accessible, hit_dists, bbox = run_gpu_ray_casting(
        gmsh_mesh, pts, directions, tool_radius=args.tool_radius, near_tol=args.near_tol)
    
    exit_lengths = compute_bbox_exit_lengths(pts, directions, bbox)

    # Exit length metrics averaged along line sample points
    avg_lengths_per_dir = exit_lengths.mean(axis=0)
    min_idx = np.argmin(avg_lengths_per_dir)
    max_idx = np.argmax(avg_lengths_per_dir)

    min_dir = directions[min_idx]
    max_dir = directions[max_idx]

    # ── Terminal Output ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"WELD LINE RAY CAST ANALYSIS SUMMARY (LINE INDEX: {selected_idx})")
    print("="*70)
    print(f"Selected Weld Line Index        : {selected_idx} (out of {n_concave} total lines)")
    print(f"Generated Parametric Directions : {len(directions)} rays (bounded within face planes)")
    print(f"Tested Weld Line Avg Exit Ray Lengths (across {len(pts)} edge points):")
    print(f"  Shortest Exit Ray Length : {avg_lengths_per_dir[min_idx]:.2f} mm")
    print(f"    Angle Direction Vector : [{min_dir[0]:.4f}, {min_dir[1]:.4f}, {min_dir[2]:.4f}]")
    print(f"  Longest Exit Ray Length  : {avg_lengths_per_dir[max_idx]:.2f} mm")
    print(f"    Angle Direction Vector : [{max_dir[0]:.4f}, {max_dir[1]:.4f}, {max_dir[2]:.4f}]")
    print("="*70 + "\n")

    # ── Vis 3: Unobstructed vs Obstructed Torch Rays ─────────────────────────
    pl3 = pv.Plotter(window_size=[1000, 800])
    pl3.add_mesh(gmsh_mesh, color='lightgray', opacity=0.6)
    
    for i, p in enumerate(pts):
        for j, d in enumerate(directions):
            is_clear = accessible[i, j]
            bbox_exit_dist = exit_lengths[i, j]
            
            if is_clear:
                draw_len = bbox_exit_dist
                color = 'green'
            else:
                draw_len = min(hit_dists[i, j], bbox_exit_dist)
                color = 'red'
                
            endpoint = p + d * draw_len
            pl3.add_lines(np.array([p, endpoint]), color=color, width=1.5)

    pl3.add_title(
        f"3. Weld Line {selected_idx} Ray Casting (Green = Clear, Red = Stopped at Hit | Radius = {args.tool_radius}mm, Near Tol = {args.near_tol}mm)", 
        font_size=12
    )
    pl3.show()


if __name__ == '__main__':
    main()