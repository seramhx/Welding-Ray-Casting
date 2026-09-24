import os
import json
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import warp as wp
import gmsh

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomAbs import GeomAbs_SurfaceType, GeomAbs_Cylinder
from OCC.Extend.TopologyUtils import TopologyExplorer



def load_step(step_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise FileNotFoundError(f"Could not read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()



def mesh_occ_shape_with_gmsh(occ_shape, minh=0.5, maxh=3.0, curvature=10):
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


def _mesh_cache_paths(step_path):
    base = Path(step_path)
    return base.with_suffix(".stl"), base.with_suffix(".facemap.json")


def build_face_tagged_mesh(step_path, faces, minh=0.5, maxh=3.0, curvature=10, force_remesh=False):
    stl_path, map_path = _mesh_cache_paths(step_path)
    step_mtime = os.path.getmtime(step_path)
    mesh_params = {"minh": minh, "maxh": maxh, "curvature": curvature}

    if not force_remesh and stl_path.exists() and map_path.exists():
        with open(map_path, "r") as f:
            mapping = json.load(f)
        if (mapping.get("step_mtime") == step_mtime and
                mapping.get("num_faces") == len(faces) and
                mapping.get("mesh_params") == mesh_params):
            print(f"Using cached mesh: {stl_path.name} (+ {map_path.name})")
            mesh = pv.read(str(stl_path))
            face_ids = np.zeros(mesh.n_cells, dtype=np.int32)
            for entry in mapping["faces"]:
                s, c = entry["triangle_start"], entry["triangle_count"]
                face_ids[s:s + c] = entry["face_index"]
            mesh.cell_data["FaceID"] = face_ids
            return mesh
        print("Cached mesh is stale (STEP file or mesh settings changed) - regenerating.")

    print(f"Meshing {len(faces)} STEP faces via Gmsh (this can take a while for complex parts)...")
    all_points, all_tris, face_entries = [], [], []
    offset_pts, offset_tris = 0, 0

    for idx, face in enumerate(faces):
        face_pv = mesh_occ_shape_with_gmsh(face, minh, maxh, curvature)
        if face_pv is None:
            continue
        pts = face_pv.points
        tris = face_pv.faces.reshape(-1, 4)[:, 1:]
        all_points.append(pts)
        all_tris.append(tris + offset_pts)
        face_entries.append({"face_index": idx, "triangle_start": offset_tris, "triangle_count": len(tris)})
        offset_pts += len(pts)
        offset_tris += len(tris)

    points = np.vstack(all_points).astype(np.float32)
    tris = np.vstack(all_tris)
    faces_fmt = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).flatten()

    mesh = pv.PolyData(points, faces_fmt)
    face_ids = np.zeros(len(tris), dtype=np.int32)
    for entry in face_entries:
        s, c = entry["triangle_start"], entry["triangle_count"]
        face_ids[s:s + c] = entry["face_index"]
    mesh.cell_data["FaceID"] = face_ids

    mesh.save(str(stl_path))
    with open(map_path, "w") as f:
        json.dump({
            "step_path": str(step_path),
            "step_mtime": step_mtime,
            "num_faces": len(faces),
            "mesh_params": mesh_params,
            "faces": face_entries,
        }, f, indent=2)
    print(f"Cached mesh written to {stl_path.name} (+ {map_path.name})")

    return mesh



def find_cylinder_face_groups(faces, radius_tol=1e-3, axis_tol=1e-3, component_of_face=None):
    cyl_entries = []
    for idx, f in enumerate(faces):
        adaptor = BRepAdaptor_Surface(f, True)
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue
        cyl = adaptor.Cylinder()
        ax = cyl.Axis()
        loc = ax.Location()
        d = ax.Direction()
        cyl_entries.append((
            idx,
            np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64),
            np.array([d.X(), d.Y(), d.Z()], dtype=np.float64),
            cyl.Radius(),
        ))

    groups = []
    used = set()
    for i in range(len(cyl_entries)):
        idx_i, loc_i, dir_i, r_i = cyl_entries[i]
        if idx_i in used:
            continue
        group = [idx_i]
        used.add(idx_i)
        for j in range(i + 1, len(cyl_entries)):
            idx_j, loc_j, dir_j, r_j = cyl_entries[j]
            if idx_j in used:
                continue
            if component_of_face is not None and component_of_face[idx_i] != component_of_face[idx_j]:
                continue
            if abs(r_i - r_j) > radius_tol:
                continue
            if abs(abs(np.dot(dir_i, dir_j)) - 1.0) > 1e-4:
                continue
            offset = loc_j - loc_i
            perp = offset - np.dot(offset, dir_i) * dir_i
            if np.linalg.norm(perp) > axis_tol:
                continue
            group.append(idx_j)
            used.add(idx_j)
        groups.append(group)

    face_to_group = {}
    for group in groups:
        for idx in group:
            face_to_group[idx] = group
    for idx in range(len(faces)):
        face_to_group.setdefault(idx, [idx])
    return face_to_group


def report_cylinder_groups(face_groups):
    multi = sorted({tuple(sorted(g)) for g in face_groups.values() if len(g) > 1})
    if multi:
        print(f"Detected {len(multi)} split-cylinder face group(s) (same axis+radius, "
              f"stored as separate STEP faces):")
        for g in multi:
            print(f"  faces {list(g)}")
    else:
        print("No split-cylinder face groups detected.")
    return multi



def component_id_per_face(shape, faces):
    """For each face in `faces` (as returned by TopologyExplorer(shape).faces()), finds which
    solid of the shape it belongs to, by identity against each solid's own face list. Returns
    (component_of_face, n_solids). Used to retrofit component coloring onto scripts that load
    faces as one flat list across the whole shape without already tracking this."""
    solids = list(TopologyExplorer(shape).solids())
    component_of_face = [-1] * len(faces)
    for comp_idx, solid in enumerate(solids):
        for sf in TopologyExplorer(solid).faces():
            for i, f in enumerate(faces):
                if component_of_face[i] == -1 and f.IsSame(sf):
                    component_of_face[i] = comp_idx
                    break
    return component_of_face, len(solids)


def tag_component_ids(tagged_mesh, component_of_face):
    lookup = np.array(component_of_face, dtype=np.int32)
    tagged_mesh.cell_data["ComponentID"] = lookup[tagged_mesh.cell_data["FaceID"]]
    return tagged_mesh


def add_scene_mesh(pl, mesh, fallback_color="lightgray", **kwargs):
    """Adds mesh as the scene's background/context, colored per assembly component
    (ComponentID cell data, see tag_component_ids) if present, so an assembly's separate
    components are visually distinguishable instead of one flat color -- or `fallback_color`
    for a single-body part, where every face belongs to the one same component and per-
    component coloring wouldn't show anything."""
    if "ComponentID" in mesh.cell_data:
        pl.add_mesh(mesh, scalars="ComponentID", cmap="tab10", show_scalar_bar=False, **kwargs)
    else:
        pl.add_mesh(mesh, color=fallback_color, **kwargs)


def pick_two_faces(mesh, faces, face_groups=None, base_scalars=None, base_cmap="tab10"):
    picked_groups = []
    pl = pv.Plotter(window_size=[1100, 850])
    if base_scalars is None and "ComponentID" in mesh.cell_data:
        base_scalars, base_cmap = "ComponentID", "tab10"
    if base_scalars is not None:
        pl.add_mesh(mesh, scalars=base_scalars, cmap=base_cmap, show_edges=True, edge_color="dimgray",
                    show_scalar_bar=False)
    else:
        pl.add_mesh(mesh, color="lightgray", show_edges=True, edge_color="dimgray")
    pl.add_text(
        "Left-click two adjacent faces to select the weld joint, then close the window.",
        font_size=11, color="black"
    )

    highlight_colors = ["red", "blue"]

    def on_pick(picked):
        if picked is None or picked.n_cells == 0:
            return
        fid = int(picked.cell_data["FaceID"][0])
        group = face_groups.get(fid, [fid]) if face_groups is not None else [fid]
        if any(fid in g for g in picked_groups):
            print(f"Face {fid} already selected, ignoring duplicate pick.")
            return
        if len(picked_groups) >= 2:
            print("Two faces already selected. Close the window to continue.")
            return
        picked_groups.append(group)
        sub = mesh.extract_cells(np.isin(mesh.cell_data["FaceID"], group))
        pl.add_mesh(sub, color=highlight_colors[len(picked_groups) - 1], name=f"picked_{len(picked_groups)}")
        if len(group) > 1:
            print(f"Picked face #{len(picked_groups)}: FaceID={fid} "
                  f"(grouped with faces {sorted(group)} -- same cylindrical surface split "
                  f"across multiple STEP faces)")
        else:
            print(f"Picked face #{len(picked_groups)}: FaceID={fid}")
        if len(picked_groups) == 2:
            print("Two faces selected. Close the window to continue.")

    pl.enable_element_picking(callback=on_pick, mode="cell", left_clicking=True, show_message=False)
    pl.show()

    if len(picked_groups) < 2:
        raise RuntimeError("Two faces were not selected. Re-run and pick exactly two adjacent faces.")

    group1, group2 = picked_groups[0], picked_groups[1]
    return [faces[i] for i in group1], [faces[i] for i in group2], group1, group2



def find_shared_edges(topo, faces1, faces2):
    faces1 = faces1 if isinstance(faces1, (list, tuple)) else [faces1]
    faces2 = faces2 if isinstance(faces2, (list, tuple)) else [faces2]
    edges1 = [e for f in faces1 for e in topo.edges_from_face(f)]
    edges2 = [e for f in faces2 for e in topo.edges_from_face(f)]
    shared = []
    for e1 in edges1:
        for e2 in edges2:
            if e1.IsSame(e2) and not any(e1.IsSame(s) for s in shared):
                shared.append(e1)
    return shared


def _edge_endpoints(edge):
    adaptor = BRepAdaptor_Curve(edge)
    p0 = adaptor.Value(adaptor.FirstParameter())
    p1 = adaptor.Value(adaptor.LastParameter())
    return (np.array([p0.X(), p0.Y(), p0.Z()], dtype=np.float64),
            np.array([p1.X(), p1.Y(), p1.Z()], dtype=np.float64))


def assemble_edge_chains(edges, tol=1e-4):
    remaining = list(edges)
    endpoints = [_edge_endpoints(e) for e in remaining]

    def same_pt(a, b):
        return np.linalg.norm(a - b) < tol

    chains = []
    while remaining:
        e0 = remaining.pop(0)
        p_start, cur_end = endpoints.pop(0)
        chain = [(e0, False)]
        changed = True
        while changed:
            changed = False
            for i in range(len(remaining)):
                a, b = endpoints[i]
                if same_pt(a, cur_end):
                    chain.append((remaining.pop(i), False))
                    endpoints.pop(i)
                    cur_end = b
                    changed = True
                    break
                if same_pt(b, cur_end):
                    chain.append((remaining.pop(i), True))
                    endpoints.pop(i)
                    cur_end = a
                    changed = True
                    break
        chains.append((chain, same_pt(cur_end, p_start)))
    return chains


def _sample_edge_polyline(edge, n=30):
    adaptor = BRepAdaptor_Curve(edge)
    u_min, u_max = adaptor.FirstParameter(), adaptor.LastParameter()
    return np.array(
        [[adaptor.Value(p).X(), adaptor.Value(p).Y(), adaptor.Value(p).Z()] for p in np.linspace(u_min, u_max, n)],
        dtype=np.float32
    )


def _sample_chain_polyline(chain, n_per_edge=30):
    segments = []
    for i, (edge, reversed_) in enumerate(chain):
        seg = _sample_edge_polyline(edge, n_per_edge)
        if reversed_:
            seg = seg[::-1]
        segments.append(seg if i == 0 else seg[1:])
    return np.vstack(segments)


def pick_edge_by_index(mesh, chains, title, chain_colors=None):
    pl = pv.Plotter(window_size=[1000, 800])
    add_scene_mesh(pl, mesh, fallback_color="whitesmoke", opacity=0.5)
    palette = ["magenta", "cyan", "yellow", "orange", "lime", "purple"]

    for i, chain in enumerate(chains):
        pts = _sample_chain_polyline(chain)
        color = chain_colors[i] if chain_colors is not None else palette[i % len(palette)]
        pl.add_lines(pts, color=color, width=5, connected=True, label=f"Candidate {i} ({len(chain)} piece(s))")
        pl.add_point_labels([pts[len(pts) // 2]], [str(i)], font_size=20, text_color=color, shape=None)

    pl.add_legend()
    pl.add_title(title, font_size=11)
    pl.show()

    choice = -1
    while not (0 <= choice < len(chains)):
        raw = input(f"Enter index of weld edge/loop to use [0-{len(chains) - 1}]: ").strip()
        if raw.isdigit():
            choice = int(raw)
    return chains[choice]


def resolve_weld_seam(topo, tagged_mesh, face1, face2, allow_manual_pick=False):
    shared_edges = find_shared_edges(topo, face1, face2)
    if not shared_edges:
        if allow_manual_pick:
            return None
        raise RuntimeError(
            "The two selected faces do not share a B-Rep edge. If this STEP file is an assembly "
            "whose components don't touch topologically, use weldfinal_assembly.py instead."
        )

    chains = assemble_edge_chains(shared_edges)
    closed_chains = [c for c, is_closed in chains if is_closed]

    if len(chains) == 1:
        chain, is_closed = chains[0]
        if is_closed and len(chain) > 1:
            print(f"Assembled a closed weld loop from {len(chain)} edge piece(s) "
                  f"(STEP split-surface artifact bridged automatically).")
        return chain

    if len(closed_chains) == 1:
        print(f"Found {len(chains)} shared-edge piece(s); assembled a single closed weld loop from "
              f"{len(closed_chains[0])} of them (STEP split-surface artifact bridged automatically).")
        return closed_chains[0]

    print(f"Found {len(chains)} disconnected weld-seam candidate(s) between the selected faces.")
    return pick_edge_by_index(
        tagged_mesh, [c for c, _ in chains],
        "Multiple weld-seam candidates found - note the index, then close the window")



def _face_group_owned_edges(face_or_faces):
    faces = face_or_faces if isinstance(face_or_faces, (list, tuple)) else [face_or_faces]
    return {id(f): list(TopologyExplorer(f).edges()) for f in faces}


def _pcurve_uv(edge, face_or_faces, u_param, owned_edges_cache=None):
    """Returns the edge's pcurve UV on whichever face in the group truly, topologically owns
    it -- or (None, None) if none do. BRep_Tool.CurveOnSurface(edge, face) is NOT a safe way to
    test ownership by itself: OCC returns a non-null Geom2d_Curve even for a face the edge does
    not belong to at all (verified empirically -- it appears to compute an on-demand projection
    rather than refusing), so calling it on an unrelated face can silently yield a plausible-
    looking but meaningless UV, which is only harmless on a plane (any UV maps to the same
    normal) and gives a wrong, sometimes near-constant normal on a curved surface like a
    cylinder. True ownership is instead checked directly via the face's own topology."""
    faces = face_or_faces if isinstance(face_or_faces, (list, tuple)) else [face_or_faces]
    if owned_edges_cache is None:
        owned_edges_cache = _face_group_owned_edges(faces)

    def is_true_owner(f):
        return any(edge.IsSame(e) for e in owned_edges_cache.get(id(f), []))

    for face in faces:
        if not is_true_owner(face):
            continue
        try:
            curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
        except Exception:
            continue
        if curve2d is None:
            continue
        p2d = curve2d.Value(u_param)
        return (p2d.X(), p2d.Y()), face
    return None, None


def _surface_point_and_normal(face, uv, fallback_pnt):
    surf_handle = BRep_Tool.Surface(face)
    if uv is None:
        sas = ShapeAnalysis_Surface(surf_handle)
        uv_pnt = sas.ValueOfUV(fallback_pnt, 1e-3)
        u, v = uv_pnt.Coord(1), uv_pnt.Coord(2)
    else:
        u, v = uv

    surf_pt = surf_handle.Value(u, v)
    point = np.array([surf_pt.X(), surf_pt.Y(), surf_pt.Z()], dtype=np.float32)

    adaptor = BRepAdaptor_Surface(face, True)
    props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 1, 1e-4)
    normal_defined = bool(props.IsNormalDefined())
    if normal_defined:
        n = props.Normal()
        # adaptor.Surface().Surface() is the RAW underlying geometry, in the surface's own local
        # frame, ignoring the face's placement -- fine for a shape with an identity location (a
        # lone STEP part), but silently wrong wherever the face belongs to a rotated instance
        # within an assembly (BRep_Tool.Surface(face)/surf_handle.Value(u, v) above DO account
        # for it, which is why the projected 3D point is correct even when this raw normal isn't).
        # The face's own placement transform must be applied to the normal the same way.
        n.Transform(face.Location().Transformation())
        vec = np.array([n.X(), n.Y(), n.Z()], dtype=np.float32)
        if face.Orientation() == TopAbs_REVERSED:
            vec = -vec
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return point, vec, normal_defined


def get_face_normal_at_point_brep(face, pnt):
    _, vec, _ = _surface_point_and_normal(face, None, pnt)
    return vec


def face_surface_type_str(face):
    adaptor = BRepAdaptor_Surface(face, True)
    return GeomAbs_SurfaceType(adaptor.GetType()).name.replace("GeomAbs_", "")


def build_face_normal_mesh(tagged_mesh, face_idx_or_indices):
    indices = face_idx_or_indices if isinstance(face_idx_or_indices, (list, tuple)) else [face_idx_or_indices]
    mask = np.isin(tagged_mesh.cell_data["FaceID"], indices)
    face_pv = tagged_mesh.extract_cells(mask).extract_surface()
    return face_pv.compute_normals(
        cell_normals=True, point_normals=False, consistent_normals=False, auto_orient_normals=False)


def build_per_face_normal_meshes(tagged_mesh, face_or_faces, idx_or_indices):
    faces_list = face_or_faces if isinstance(face_or_faces, (list, tuple)) else [face_or_faces]
    idx_list = idx_or_indices if isinstance(idx_or_indices, (list, tuple)) else [idx_or_indices]
    if len(faces_list) < 2:
        return {}
    per_face = {}
    for i, face in enumerate(faces_list):
        sibling_indices = [j for k, j in enumerate(idx_list) if k != i]
        per_face[id(face)] = build_face_normal_mesh(tagged_mesh, sibling_indices)
    return per_face


def _mesh_facet_point_and_normal(face_normal_mesh, gp_p):
    query = np.array([gp_p.X(), gp_p.Y(), gp_p.Z()], dtype=np.float32)
    cell_id = face_normal_mesh.find_closest_cell(query)
    cell_pts = face_normal_mesh.extract_cells([cell_id]).points
    point = np.array(cell_pts, dtype=np.float32).mean(axis=0)
    normal = np.array(face_normal_mesh.cell_data["Normals"][cell_id], dtype=np.float32)
    return point, normal


def _edge_length_estimate(edge, n=20):
    adaptor = BRepAdaptor_Curve(edge)
    u0, u1 = adaptor.FirstParameter(), adaptor.LastParameter()
    pts = np.array([[adaptor.Value(u).X(), adaptor.Value(u).Y(), adaptor.Value(u).Z()]
                     for u in np.linspace(u0, u1, n)], dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def sample_edge_with_bisector(chain, face1, face2, num_samples=15, mesh_fallback1=None, mesh_fallback2=None):
    lengths = [_edge_length_estimate(edge) for edge, _ in chain]
    total_length = sum(lengths) or 1.0
    counts = [max(2, round(num_samples * l / total_length)) for l in lengths]

    owned1 = _face_group_owned_edges(face1)
    owned2 = _face_group_owned_edges(face2)

    pts, tangents, normals1, normals2, bisectors = [], [], [], [], []
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
            t_vec = np.array([gp_v.X(), gp_v.Y(), gp_v.Z()], dtype=np.float32)
            if reversed_:
                t_vec = -t_vec
            t_norm = np.linalg.norm(t_vec)
            t_vec = t_vec / t_norm if t_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

            uv1, owner1 = _pcurve_uv(edge, face1, p, owned_edges_cache=owned1)
            uv2, owner2 = _pcurve_uv(edge, face2, p, owned_edges_cache=owned2)

            if uv1 is not None:
                _, n1, _ = _surface_point_and_normal(owner1, uv1, gp_p)
                oid1 = id(owner1)
            elif mesh_fallback1 is not None:
                _, n1 = _mesh_facet_point_and_normal(mesh_fallback1, gp_p)
                oid1 = None
            else:
                fallback_face1 = face1[0] if isinstance(face1, (list, tuple)) else face1
                _, n1, _ = _surface_point_and_normal(fallback_face1, None, gp_p)
                oid1 = None

            if uv2 is not None:
                _, n2, _ = _surface_point_and_normal(owner2, uv2, gp_p)
                oid2 = id(owner2)
            elif mesh_fallback2 is not None:
                _, n2 = _mesh_facet_point_and_normal(mesh_fallback2, gp_p)
                oid2 = None
            else:
                fallback_face2 = face2[0] if isinstance(face2, (list, tuple)) else face2
                _, n2, _ = _surface_point_and_normal(fallback_face2, None, gp_p)
                oid2 = None

            bis = n1 + n2
            b_norm = np.linalg.norm(bis)
            bis = bis / b_norm if b_norm > 1e-6 else n1

            pts.append(p3d)
            tangents.append(t_vec)
            normals1.append(n1)
            normals2.append(n2)
            bisectors.append(bis)
            owner1_ids.append(oid1)
            owner2_ids.append(oid2)

    return (np.array(pts, dtype=np.float32),
            np.array(normals1, dtype=np.float32),
            np.array(normals2, dtype=np.float32),
            np.array(bisectors, dtype=np.float32),
            np.array(tangents, dtype=np.float32),
            owner1_ids,
            owner2_ids)



@wp.kernel
def _ray_cast_kernel(
    mesh       : wp.uint64,
    origins    : wp.array(dtype=wp.vec3),
    directions : wp.array(dtype=wp.vec3),
    accessible : wp.array(dtype=int),
    hit_dists  : wp.array(dtype=float),
    max_dist   : float,
    near_tol   : float
):
    tid = wp.tid()
    ray_origin = origins[tid] + directions[tid] * near_tol
    ray_dir = directions[tid]

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)

    if wp.mesh_query_ray(mesh, ray_origin, ray_dir, max_dist, t, u, v, sign, normal, face):
        accessible[tid] = 0
        hit_dists[tid] = t + near_tol
    else:
        accessible[tid] = 1
        hit_dists[tid] = max_dist


def run_gpu_ray_casting(obstruction_mesh, origins, directions, max_dist, near_tol=0.5):
    wp.init()

    faces = obstruction_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32).flatten()
    verts_wp = wp.array(obstruction_mesh.points.astype(np.float32), dtype=wp.vec3, device='cuda')
    faces_wp = wp.array(faces, dtype=int, device='cuda')
    wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp)

    orig_wp = wp.array(origins.astype(np.float32), dtype=wp.vec3, device='cuda')
    dirs_wp = wp.array(directions.astype(np.float32), dtype=wp.vec3, device='cuda')
    acc_wp = wp.zeros(len(origins), dtype=int, device='cuda')
    hit_wp = wp.zeros(len(origins), dtype=float, device='cuda')

    wp.launch(
        kernel=_ray_cast_kernel,
        dim=len(origins),
        inputs=[wp_mesh.id, orig_wp, dirs_wp, acc_wp, hit_wp, float(max_dist), float(near_tol)],
        device='cuda'
    )

    return acc_wp.numpy().astype(bool), hit_wp.numpy()


def bbox_diagonal(mesh):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))



def compute_ring_perpendicular_basis(directions):
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(directions), 1))
    parallel_mask = np.abs(np.sum(directions * ref, axis=1)) > 0.99
    ref[parallel_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    u = np.cross(directions, ref)
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    u_norm[u_norm < 1e-9] = 1.0
    u = u / u_norm

    v = np.cross(directions, u)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm[v_norm < 1e-9] = 1.0
    v = v / v_norm

    return u, v


def compute_clearance_ring_rays(points, directions, n_rings=8, clearance_radius=5.0, ring_offset=None):
    if ring_offset is None:
        ring_offset = clearance_radius

    u, v = compute_ring_perpendicular_basis(directions)
    forward_origin = points + directions * ring_offset

    ring_origins = np.empty((n_rings, len(points), 3), dtype=np.float32)
    ring_dirs = np.empty((n_rings, len(points), 3), dtype=np.float32)
    for i in range(n_rings):
        angle = 2.0 * np.pi * i / n_rings
        offset = clearance_radius * (np.cos(angle) * u + np.sin(angle) * v)
        ring_origins[i] = forward_origin + offset
        ring_dirs[i] = directions
    return ring_origins, ring_dirs


def cast_ray_bundle_with_clearance(tagged_mesh, pts, directions, max_dist, near_tol,
                                    n_rings, clearance_radius, ring_offset=None):
    center_accessible, center_hit_dists = run_gpu_ray_casting(
        tagged_mesh, pts, directions, max_dist, near_tol=near_tol)

    if n_rings <= 0:
        return center_accessible, center_accessible, center_hit_dists, None, None, None, None

    ring_origins, ring_dirs = compute_clearance_ring_rays(
        pts, directions, n_rings, clearance_radius, ring_offset=ring_offset)
    n_pts = len(pts)
    ring_accessible_flat, ring_hit_flat = run_gpu_ray_casting(
        tagged_mesh, ring_origins.reshape(-1, 3), ring_dirs.reshape(-1, 3), max_dist, near_tol=near_tol)

    ring_accessible = ring_accessible_flat.reshape(n_rings, n_pts)
    ring_hit_dists = ring_hit_flat.reshape(n_rings, n_pts)

    combined_accessible = center_accessible & np.all(ring_accessible, axis=0)
    return combined_accessible, center_accessible, center_hit_dists, ring_accessible, ring_hit_dists, ring_origins, ring_dirs


def prompt_for_clearance_rings(default_n_rings=8, default_clearance_radius=5.0):
    answer = input("Simulate a spherical tool via clearance ring rays around each ray? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return 0, default_clearance_radius, None

    n_raw = input(f"  Number of ring rays per point [{default_n_rings}]: ").strip()
    n_rings = int(n_raw) if n_raw else default_n_rings

    r_raw = input(f"  Clearance radius in mm [{default_clearance_radius}]: ").strip()
    clearance_radius = float(r_raw) if r_raw else default_clearance_radius

    o_raw = input(f"  Ring start-point forward offset in mm "
                  f"[blank = radius = {clearance_radius:.1f}]: ").strip()
    ring_offset = float(o_raw) if o_raw else None

    return n_rings, clearance_radius, ring_offset


def compute_group_normal_signs_by_raycast(group_mesh, pts, normals, owner_ids, per_face_meshes=None):
    votes = {}
    for pt, n, oid in zip(pts, normals, owner_ids):
        if oid is None:
            continue
        sibling_mesh = per_face_meshes.get(oid) if per_face_meshes else None
        ref_mesh = sibling_mesh if (sibling_mesh is not None and sibling_mesh.n_cells > 0) else group_mesh
        cell_id = ref_mesh.find_closest_cell(pt)
        mesh_n = np.array(ref_mesh.cell_data["Normals"][cell_id], dtype=np.float32)
        is_consistent = bool(np.dot(mesh_n, n) >= 0)
        votes.setdefault(oid, []).append(is_consistent)

    signs = {}
    for oid, vote_list in votes.items():
        n_consistent = sum(vote_list)
        n_inconsistent = len(vote_list) - n_consistent
        signs[oid] = -1.0 if n_inconsistent > n_consistent else 1.0
    return signs


def apply_normal_sign_correction(normals, owner_ids, signs):
    corrected = normals.copy()
    for i, oid in enumerate(owner_ids):
        if oid is not None and signs.get(oid, 1.0) < 0:
            corrected[i] = -corrected[i]
    return corrected


def _id_to_face_index(face_or_faces, idx_or_indices):
    if idx_or_indices is None:
        return {}
    faces_list = face_or_faces if isinstance(face_or_faces, (list, tuple)) else [face_or_faces]
    idx_list = idx_or_indices if isinstance(idx_or_indices, (list, tuple)) else [idx_or_indices]
    return {id(f): i for f, i in zip(faces_list, idx_list)}


def _concat_or_empty(arrays):
    arrays = [a for a in arrays if len(a) > 0]
    if not arrays:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def add_ray_batch(pl, origins, endpoints, color, width=1.0, label=None):
    origins = np.asarray(origins, dtype=np.float32)
    endpoints = np.asarray(endpoints, dtype=np.float32)
    n = len(origins)
    if n == 0:
        return
    points = np.empty((2 * n, 3), dtype=np.float32)
    points[0::2] = origins
    points[1::2] = endpoints
    lines = np.empty((n, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * n, 2)
    lines[:, 2] = np.arange(1, 2 * n, 2)
    poly = pv.PolyData(points, lines=lines.flatten())
    pl.add_mesh(poly, color=color, line_width=width, label=label)



def _render_bisector_scene(tagged_mesh, weld_seam, pts, bis_arr, center_accessible, center_hit_dists,
                            ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                            clearance_radius, max_dist, n_accessible, draw_cap=None):
    pl = pv.Plotter(window_size=[1100, 850])
    add_scene_mesh(pl, tagged_mesh, opacity=0.6)

    edge_pts = _sample_chain_polyline(weld_seam)
    pl.add_lines(edge_pts, color="black", width=5, connected=True, label="Selected weld seam")

    full_draw_lens = np.where(center_accessible, max_dist, center_hit_dists)
    draw_lens = full_draw_lens if draw_cap is None else np.minimum(full_draw_lens, draw_cap)
    endpoints = pts + bis_arr * draw_lens[:, None]
    add_ray_batch(pl, pts[center_accessible], endpoints[center_accessible], "green", width=2.5, label="Bisector clear")
    add_ray_batch(pl, pts[~center_accessible], endpoints[~center_accessible], "red", width=2.5, label="Bisector blocked")

    if n_rings > 0:
        flat_origins = ring_origins.reshape(-1, 3)
        flat_dirs = ring_dirs.reshape(-1, 3)
        flat_acc = ring_accessible.reshape(-1)
        flat_hits = ring_hit_dists.reshape(-1)
        flat_full_draw_lens = np.where(flat_acc, max_dist, flat_hits)
        flat_draw_lens = flat_full_draw_lens if draw_cap is None else np.minimum(flat_full_draw_lens, draw_cap)
        flat_endpoints = flat_origins + flat_dirs * flat_draw_lens[:, None]
        add_ray_batch(pl, flat_origins[flat_acc], flat_endpoints[flat_acc], "palegreen", width=1.0)
        add_ray_batch(pl, flat_origins[~flat_acc], flat_endpoints[~flat_acc], "lightcoral", width=1.0)

    pl.add_points(pts, color="blue", point_size=10, render_points_as_spheres=True)
    pl.add_legend()
    title = f"Torch Bisector Ray Accessibility (Green = Clear, Red = Obstructed | {n_accessible}/{len(pts)} accessible"
    title += f", {clearance_radius:.1f}mm tool clearance via {n_rings} ring rays)" if n_rings > 0 else ")"
    if draw_cap is not None:
        title += f"\nRay draw distance capped at {draw_cap:.1f} mm"
    pl.add_title(title, font_size=12)
    pl.show()


def prompt_for_second_view_draw_distance():
    answer = input("Produce a second visualization with a shorter ray draw distance? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return None
    d_raw = input("Ray draw distance in mm: ").strip()
    if not d_raw:
        return None
    try:
        return float(d_raw)
    except ValueError:
        return None


def analyze_and_visualize(tagged_mesh, weld_seam, face1, face2, face1_idx=None, face2_idx=None,
                           num_samples=15, near_tol=0.5, n_rings=0, clearance_radius=5.0, ring_offset=None):
    mesh_fallback1 = build_face_normal_mesh(tagged_mesh, face1_idx) if face1_idx is not None else None
    mesh_fallback2 = build_face_normal_mesh(tagged_mesh, face2_idx) if face2_idx is not None else None
    per_face_meshes1 = build_per_face_normal_meshes(tagged_mesh, face1, face1_idx) if face1_idx is not None else None
    per_face_meshes2 = build_per_face_normal_meshes(tagged_mesh, face2, face2_idx) if face2_idx is not None else None

    print(f"Sampling {num_samples} points along the weld seam ({len(weld_seam)} piece(s)) "
          f"and computing bisector directions...")
    pts, n1_arr, n2_arr, bis_arr, tan_arr, owner1_ids, owner2_ids = sample_edge_with_bisector(
        weld_seam, face1, face2, num_samples=num_samples,
        mesh_fallback1=mesh_fallback1, mesh_fallback2=mesh_fallback2)

    max_dist = bbox_diagonal(tagged_mesh)

    if mesh_fallback1 is not None and mesh_fallback2 is not None:
        id_to_idx = {}
        id_to_idx.update(_id_to_face_index(face1, face1_idx))
        id_to_idx.update(_id_to_face_index(face2, face2_idx))
        signs1 = compute_group_normal_signs_by_raycast(mesh_fallback1, pts, n1_arr, owner1_ids,
                                                         per_face_meshes=per_face_meshes1)
        signs2 = compute_group_normal_signs_by_raycast(mesh_fallback2, pts, n2_arr, owner2_ids,
                                                         per_face_meshes=per_face_meshes2)
        flipped = sorted({id_to_idx.get(oid, oid) for s in (signs1, signs2) for oid, v in s.items() if v < 0})
        if flipped:
            print(f"Corrected inward-pointing normal(s) via mesh-winding cross-check (majority vote "
                  f"over the seam's own sample points): faces {flipped}")
            n1_arr = apply_normal_sign_correction(n1_arr, owner1_ids, signs1)
            n2_arr = apply_normal_sign_correction(n2_arr, owner2_ids, signs2)
            bis_arr = n1_arr + n2_arr
            bis_norms = np.linalg.norm(bis_arr, axis=1, keepdims=True)
            bis_norms[bis_norms < 1e-6] = 1.0
            bis_arr = bis_arr / bis_norms

    joint_angles = np.degrees(np.arccos(np.clip(np.sum(n1_arr * n2_arr, axis=1), -1.0, 1.0)))
    print(f"Face1/Face2 normal angle along seam : min={joint_angles.min():.1f} deg, "
          f"max={joint_angles.max():.1f} deg, mean={joint_angles.mean():.1f} deg")
    if joint_angles.mean() > 165:
        print("  >>> The two faces meet almost tangentially (a smooth fillet, not a sharp corner) -- "
              "the bisector direction is close to the faces' shared tangent plane rather than pointing")
        print("      cleanly away from the part, so grazing/obstructed hits can be a genuine result here.")

    print(f"Casting rays toward the joint bisector, length = part bbox diagonal = {max_dist:.2f} mm")
    if n_rings > 0:
        effective_offset = ring_offset if ring_offset is not None else clearance_radius
        print(f"Simulating a spherical tool of radius {clearance_radius:.1f} mm via {n_rings} clearance "
              f"ring ray(s) per point (start offset {effective_offset:.1f} mm forward along the ray)")

    (combined_accessible, center_accessible, center_hit_dists,
     ring_accessible, ring_hit_dists, ring_origins, ring_dirs) = cast_ray_bundle_with_clearance(
        tagged_mesh, pts, bis_arr, max_dist, near_tol, n_rings, clearance_radius, ring_offset=ring_offset)

    n_accessible = int(combined_accessible.sum())
    print("\n" + "=" * 70)
    print("WELD JOINT ACCESSIBILITY SUMMARY")
    print("=" * 70)
    print(f"Sample points along seam : {len(pts)}")
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
        if obstructed_hits.min() < 2 * near_tol:
            print("  >>> Some hits occur just past the near_tol offset -- likely self-intersection with")
            print("      the emitting mesh itself rather than a real obstruction. Try a larger --near_tol.")
    print("=" * 70 + "\n")

    _render_bisector_scene(tagged_mesh, weld_seam, pts, bis_arr, center_accessible, center_hit_dists,
                            ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                            clearance_radius, max_dist, n_accessible)

    draw_cap = prompt_for_second_view_draw_distance()
    if draw_cap is not None:
        _render_bisector_scene(tagged_mesh, weld_seam, pts, bis_arr, center_accessible, center_hit_dists,
                                ring_accessible, ring_hit_dists, ring_origins, ring_dirs, n_rings,
                                clearance_radius, max_dist, n_accessible, draw_cap=draw_cap)



def rotate_about_axis(vectors, axis, angle_deg):
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cross = np.cross(axis, vectors)
    dot = np.sum(axis * vectors, axis=1, keepdims=True)
    return vectors * cos_t + cross * sin_t + axis * dot * (1 - cos_t)


def compute_angular_tolerance_rays(bisectors, tangents, tol_deg, mode, angle_step=None):
    if mode == "workangle":
        axis = tangents
    elif mode == "travelangle":
        axis = np.cross(tangents, bisectors)
        norm = np.linalg.norm(axis, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1.0
        axis = axis / norm
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'workangle' or 'travelangle')")

    if angle_step is None or angle_step <= 0:
        angle_list = [0.0, -float(tol_deg), float(tol_deg)]
    else:
        steps = np.arange(0.0, tol_deg + 1e-9, angle_step)
        angle_list = sorted({round(float(s), 6) for s in steps} | {round(float(-s), 6) for s in steps})

    rays = {}
    for angle in angle_list:
        rays[float(angle)] = bisectors if angle == 0.0 else rotate_about_axis(bisectors, axis, angle)
    return rays


def _render_tolerance_scene(tagged_mesh, weld_seam, pts, angles_sorted, center_by_angle, center_hits_by_angle,
                             ray_dirs_by_angle, ring_acc_by_angle, ring_hits_by_angle, ring_origins_by_angle,
                             ring_dirs_by_angle, n_rings, clearance_radius, max_dist, all_accessible_mask,
                             n_pts, tol_deg, mode, draw_cap=None):
    pl = pv.Plotter(window_size=[1100, 850])
    add_scene_mesh(pl, tagged_mesh, opacity=0.6)

    edge_pts = _sample_chain_polyline(weld_seam)
    pl.add_lines(edge_pts, color="black", width=5, connected=True, label="Selected weld seam")

    line_widths = {0.0: 2.5}
    colors = {0.0: ("green", "red")}
    for a in angles_sorted:
        if a != 0.0:
            line_widths[a] = 1.5
            colors[a] = ("seagreen", "salmon")

    batches = {}
    ring_clear_o, ring_clear_e, ring_blocked_o, ring_blocked_e = [], [], [], []

    for a in angles_sorted:
        acc, hits, dirs = center_by_angle[a], center_hits_by_angle[a], ray_dirs_by_angle[a]
        clear_color, blocked_color = colors[a]
        width = line_widths[a]
        full_draw_lens = np.where(acc, max_dist, hits)
        draw_lens = full_draw_lens if draw_cap is None else np.minimum(full_draw_lens, draw_cap)
        endpoints = pts + dirs * draw_lens[:, None]

        key_clear = (clear_color, width)
        key_blocked = (blocked_color, width)
        batches.setdefault(key_clear, ([], []))
        batches.setdefault(key_blocked, ([], []))
        batches[key_clear][0].append(pts[acc])
        batches[key_clear][1].append(endpoints[acc])
        batches[key_blocked][0].append(pts[~acc])
        batches[key_blocked][1].append(endpoints[~acc])

        if n_rings > 0:
            r_acc, r_hits = ring_acc_by_angle[a], ring_hits_by_angle[a]
            r_origins, r_dirs = ring_origins_by_angle[a], ring_dirs_by_angle[a]
            flat_acc = r_acc.reshape(-1)
            flat_hits = r_hits.reshape(-1)
            flat_origins = r_origins.reshape(-1, 3)
            flat_dirs = r_dirs.reshape(-1, 3)
            flat_full_draw_lens = np.where(flat_acc, max_dist, flat_hits)
            flat_draw_lens = flat_full_draw_lens if draw_cap is None else np.minimum(flat_full_draw_lens, draw_cap)
            flat_endpoints = flat_origins + flat_dirs * flat_draw_lens[:, None]
            ring_clear_o.append(flat_origins[flat_acc])
            ring_clear_e.append(flat_endpoints[flat_acc])
            ring_blocked_o.append(flat_origins[~flat_acc])
            ring_blocked_e.append(flat_endpoints[~flat_acc])

    labels = {("green", 2.5): "Bisector clear", ("red", 2.5): "Bisector blocked",
              ("seagreen", 1.5): "Tolerance clear", ("salmon", 1.5): "Tolerance blocked"}
    for (color, width), (o_list, e_list) in batches.items():
        add_ray_batch(pl, _concat_or_empty(o_list), _concat_or_empty(e_list), color, width=width,
                      label=labels.get((color, width)))

    if n_rings > 0:
        add_ray_batch(pl, _concat_or_empty(ring_clear_o), _concat_or_empty(ring_clear_e), "palegreen", width=0.75)
        add_ray_batch(pl, _concat_or_empty(ring_blocked_o), _concat_or_empty(ring_blocked_e), "lightcoral", width=0.75)

    pl.add_points(pts, color="blue", point_size=10, render_points_as_spheres=True)
    pl.add_legend()
    axis_title = "Work-Angle" if mode == "workangle" else "Travel-Angle"
    subtitle = "(bright green/red = bisector ray, pale green/red = tolerance rays"
    subtitle += f", faint = {clearance_radius:.1f}mm clearance rings" if n_rings > 0 else ""
    subtitle += f" | {int(all_accessible_mask.sum())}/{n_pts} clear at every angle)"
    if draw_cap is not None:
        subtitle += f"\nRay draw distance capped at {draw_cap:.1f} mm"
    pl.add_title(
        f"Torch Bisector +/-{tol_deg:.0f} deg {axis_title} Tolerance Accessibility\n{subtitle}",
        font_size=11
    )
    pl.show()


def analyze_and_visualize_with_tolerance(tagged_mesh, weld_seam, face1, face2, face1_idx=None, face2_idx=None,
                                          num_samples=15, near_tol=0.5, tol_deg=10.0, mode="workangle",
                                          n_rings=0, clearance_radius=5.0, ring_offset=None, angle_step=None):
    mesh_fallback1 = build_face_normal_mesh(tagged_mesh, face1_idx) if face1_idx is not None else None
    mesh_fallback2 = build_face_normal_mesh(tagged_mesh, face2_idx) if face2_idx is not None else None
    per_face_meshes1 = build_per_face_normal_meshes(tagged_mesh, face1, face1_idx) if face1_idx is not None else None
    per_face_meshes2 = build_per_face_normal_meshes(tagged_mesh, face2, face2_idx) if face2_idx is not None else None

    print(f"Sampling {num_samples} points along the weld seam ({len(weld_seam)} piece(s)) "
          f"and computing bisector directions...")
    pts, n1_arr, n2_arr, bis_arr, tan_arr, owner1_ids, owner2_ids = sample_edge_with_bisector(
        weld_seam, face1, face2, num_samples=num_samples,
        mesh_fallback1=mesh_fallback1, mesh_fallback2=mesh_fallback2)

    max_dist = bbox_diagonal(tagged_mesh)

    if mesh_fallback1 is not None and mesh_fallback2 is not None:
        id_to_idx = {}
        id_to_idx.update(_id_to_face_index(face1, face1_idx))
        id_to_idx.update(_id_to_face_index(face2, face2_idx))
        signs1 = compute_group_normal_signs_by_raycast(mesh_fallback1, pts, n1_arr, owner1_ids,
                                                         per_face_meshes=per_face_meshes1)
        signs2 = compute_group_normal_signs_by_raycast(mesh_fallback2, pts, n2_arr, owner2_ids,
                                                         per_face_meshes=per_face_meshes2)
        flipped = sorted({id_to_idx.get(oid, oid) for s in (signs1, signs2) for oid, v in s.items() if v < 0})
        if flipped:
            print(f"Corrected inward-pointing normal(s) via mesh-winding cross-check (majority vote "
                  f"over the seam's own sample points): faces {flipped}")
            n1_arr = apply_normal_sign_correction(n1_arr, owner1_ids, signs1)
            n2_arr = apply_normal_sign_correction(n2_arr, owner2_ids, signs2)
            bis_arr = n1_arr + n2_arr
            bis_norms = np.linalg.norm(bis_arr, axis=1, keepdims=True)
            bis_norms[bis_norms < 1e-6] = 1.0
            bis_arr = bis_arr / bis_norms

    joint_angles = np.degrees(np.arccos(np.clip(np.sum(n1_arr * n2_arr, axis=1), -1.0, 1.0)))
    print(f"Face1/Face2 normal angle along seam : min={joint_angles.min():.1f} deg, "
          f"max={joint_angles.max():.1f} deg, mean={joint_angles.mean():.1f} deg")
    if joint_angles.mean() > 165:
        print("  >>> The two faces meet almost tangentially (a smooth fillet, not a sharp corner) -- "
              "the bisector direction is close to the faces' shared tangent plane rather than pointing")
        print("      cleanly away from the part, so grazing/obstructed hits can be a genuine result here.")

    axis_label = ("the joint cross-section, tilting toward each flanking face (work angle)" if mode == "workangle"
                  else "the direction of travel along the seam, push/drag tilt (travel angle)")
    sweep_desc = (f"+/-{tol_deg:.0f} deg tolerance rays" if angle_step is None or angle_step <= 0
                  else f"a +/-{tol_deg:.0f} deg sweep every {angle_step:.2g} deg")
    print(f"Casting the bisector ray plus {sweep_desc} swept within {axis_label}, "
          f"length = part bbox diagonal = {max_dist:.2f} mm")

    if n_rings > 0:
        effective_offset = ring_offset if ring_offset is not None else clearance_radius
        print(f"Simulating a spherical tool of radius {clearance_radius:.1f} mm via {n_rings} clearance "
              f"ring ray(s) per point per angle (start offset {effective_offset:.1f} mm forward along each ray)")

    ray_dirs_by_angle = compute_angular_tolerance_rays(bis_arr, tan_arr, tol_deg, mode, angle_step=angle_step)
    angles_sorted = sorted(ray_dirs_by_angle.keys())
    n_pts = len(pts)

    bundle_by_angle = {
        a: cast_ray_bundle_with_clearance(tagged_mesh, pts, ray_dirs_by_angle[a], max_dist, near_tol,
                                           n_rings, clearance_radius, ring_offset=ring_offset)
        for a in angles_sorted
    }
    combined_by_angle = {a: bundle_by_angle[a][0] for a in angles_sorted}
    center_by_angle = {a: bundle_by_angle[a][1] for a in angles_sorted}
    center_hits_by_angle = {a: bundle_by_angle[a][2] for a in angles_sorted}
    ring_acc_by_angle = {a: bundle_by_angle[a][3] for a in angles_sorted}
    ring_hits_by_angle = {a: bundle_by_angle[a][4] for a in angles_sorted}
    ring_origins_by_angle = {a: bundle_by_angle[a][5] for a in angles_sorted}
    ring_dirs_by_angle = {a: bundle_by_angle[a][6] for a in angles_sorted}

    print("\n" + "=" * 70)
    title_suffix = " + tool clearance)" if n_rings > 0 else ")"
    print("WELD JOINT ACCESSIBILITY SUMMARY (with angular tolerance" + title_suffix)
    print("=" * 70)
    print(f"Sample points along seam : {n_pts}")
    for a in angles_sorted:
        label = "Bisector (0 deg)" if a == 0.0 else f"{a:+.1f} deg"
        if n_rings > 0:
            print(f"  {label:<18}: center {int(center_by_angle[a].sum())}/{n_pts}, "
                  f"with clearance {int(combined_by_angle[a].sum())}/{n_pts}")
        else:
            print(f"  {label:<18}: {int(combined_by_angle[a].sum())} / {n_pts} accessible")

    all_accessible_mask = np.all([combined_by_angle[a] for a in angles_sorted], axis=0)
    any_blocked_mask = ~all_accessible_mask
    print(f"Points clear at every tested angle  : {int(all_accessible_mask.sum())} / {n_pts}")
    print(f"Points blocked at 1+ tested angle   : {int(any_blocked_mask.sum())} / {n_pts}")
    print("=" * 70 + "\n")

    scene_args = (tagged_mesh, weld_seam, pts, angles_sorted, center_by_angle, center_hits_by_angle,
                  ray_dirs_by_angle, ring_acc_by_angle, ring_hits_by_angle, ring_origins_by_angle,
                  ring_dirs_by_angle, n_rings, clearance_radius, max_dist, all_accessible_mask, n_pts,
                  tol_deg, mode)
    _render_tolerance_scene(*scene_args)

    draw_cap = prompt_for_second_view_draw_distance()
    if draw_cap is not None:
        _render_tolerance_scene(*scene_args, draw_cap=draw_cap)



def main():
    parser = argparse.ArgumentParser(description="Interactive Weld Joint Accessibility Pipeline")
    parser.add_argument("--step", required=True, help="Path to input STEP file")
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

    print("Loading STEP file...")
    shape = load_step(args.step)
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    print(f"Loaded part with {len(faces)} faces.")

    face_groups = find_cylinder_face_groups(faces)
    report_cylinder_groups(face_groups)

    tagged_mesh = build_face_tagged_mesh(
        args.step, faces, args.minh, args.maxh, args.curvature, force_remesh=args.force_remesh)

    print("\nOpening interactive window: left-click two adjacent faces to select the weld joint.")
    face1, face2, face1_idx, face2_idx = pick_two_faces(tagged_mesh, faces, face_groups)
    print(f"Selected faces: {face1_idx} and {face2_idx}")

    weld_seam = resolve_weld_seam(topo, tagged_mesh, face1, face2)

    analyze_and_visualize(tagged_mesh, weld_seam, face1, face2, face1_idx, face2_idx,
                          num_samples=args.num_samples, near_tol=args.near_tol,
                          n_rings=n_rings, clearance_radius=clearance_radius,
                          ring_offset=ring_offset)


if __name__ == '__main__':
    main()
