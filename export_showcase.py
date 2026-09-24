import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pyvista as pv

from weldfinal import load_step, build_face_tagged_mesh, find_cylinder_face_groups, bbox_diagonal, \
    cast_ray_bundle_with_clearance, _sample_chain_polyline, _sample_edge_polyline
from weld_line_finder import identify_components, collect_candidate_edges_assembly, \
    build_chains_from_ids, sample_and_analyze_chain, _common_and_varying_faces
from weld_sequence import weld_own_other_components, weld_label, SequenceEvaluator, \
    enumerate_all_sequences, print_ranked_summary, _score, _test_directions
from weld_sequence_from_docx import parse_weld_lines_from_docx, resolve_assembly_chain_faces_auto

THIS_DIR = Path(__file__).resolve().parent

# The two demo cases showcase.py can replay.
SAMPLE_CASES = (10, 2000)

# Rows of the ranked-orders table carried into the showcase (all 9! = 362,880 would be ~35 MB).
N_RANKING_ROWS = 25

EDGE_POLY_POINTS = 25


def export_case(num_samples, tagged_mesh, faces, face_groups, component_of_face, n_components,
                candidate_edges, own_info, specs, args):
    """Same weld-line resolution + full 9! order search as weld_sequence_from_docx.py, then the
    ray casts of every step of the optimal order -- everything showcase.py needs to redraw it."""
    print(f"\n=== {num_samples} sample points per weld line ===")
    weld_lines = []
    for i, spec in enumerate(specs):
        chains = build_chains_from_ids(spec["edge_ids"], candidate_edges)
        if len(chains) != 1:
            raise RuntimeError(f"docx line {i}: edge IDs {spec['edge_ids']} don't form one chain")
        chain = chains[0]
        other = spec["other_component"] if spec["other_component"] is not None else 0
        per_piece_faces = resolve_assembly_chain_faces_auto(
            tagged_mesh, faces, face_groups, component_of_face, chain, candidate_edges, own_info,
            args.proximity_tol, args.parallel_tol_deg, other)
        if per_piece_faces is None:
            raise RuntimeError(f"docx line {i}: couldn't resolve faces")
        with contextlib.redirect_stdout(io.StringIO()):
            pts, n1, n2, bis, tan = sample_and_analyze_chain(
                tagged_mesh, chain, per_piece_faces, faces, num_samples=num_samples)
        weld_lines.append({"docx_line": i, "edge_ids": spec["edge_ids"], "chain": chain,
                           "per_piece_faces": per_piece_faces, "points": pts, "normals1": n1,
                           "normals2": n2, "bisectors": bis, "tangents": tan})

    own_other = weld_own_other_components(weld_lines, component_of_face)
    baseline_present = set(range(n_components)) - {c for pair in own_other for c in pair}
    evaluator = SequenceEvaluator(tagged_mesh, weld_lines, own_other, baseline_present, args.near_tol)

    print(f"Evaluating all {len(weld_lines)}! weld orders...")
    results = sorted(enumerate_all_sequences(evaluator, len(weld_lines)), key=_score)
    optimal = results[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_ranked_summary(results[:N_RANKING_ROWS], own_other, optimal)
    ranking_text = (f"{len(results)} orders evaluated, top {N_RANKING_ROWS} shown:"
                    + buf.getvalue())

    max_dist = bbox_diagonal(tagged_mesh)
    data = {
        "num_samples": np.int32(num_samples),
        "n_welds": np.int32(len(weld_lines)),
        "order": np.array(optimal["order"], dtype=np.int32),
        "max_dist": np.float32(max_dist),
        "ranking_text": np.array(ranking_text),
    }
    for w, wl in enumerate(weld_lines):
        common, varying = _common_and_varying_faces(wl["per_piece_faces"])
        data[f"w{w}_label"] = np.array(weld_label(w, *own_other[w]))
        data[f"w{w}_info"] = np.array(f"docx line {wl['docx_line']}, edge ID(s) {wl['edge_ids']}")
        data[f"w{w}_seam"] = _sample_chain_polyline(wl["chain"])
        data[f"w{w}_points"] = wl["points"].astype(np.float32)
        data[f"w{w}_n1"] = wl["normals1"].astype(np.float32)
        data[f"w{w}_n2"] = wl["normals2"].astype(np.float32)
        data[f"w{w}_bis"] = wl["bisectors"].astype(np.float32)
        data[f"w{w}_common_faces"] = np.array(common, dtype=np.int32)
        data[f"w{w}_varying_faces"] = np.array(varying, dtype=np.int32)

    for s, step in enumerate(optimal["steps"]):
        wl = weld_lines[step["weld"]]
        directions = _test_directions(wl)
        mesh = evaluator.obstruction_mesh(step["present"])
        accessible, _c, hit_dists, *_ = cast_ray_bundle_with_clearance(
            mesh, wl["points"], directions, max_dist, args.near_tol, 0, 5.0)
        data[f"s{s}_present"] = np.array(step["present"], dtype=np.int32)
        data[f"s{s}_dirs"] = np.asarray(directions, dtype=np.float32)
        data[f"s{s}_accessible"] = accessible
        data[f"s{s}_hit_dists"] = hit_dists.astype(np.float32)
        print(f"  step {s + 1}: {weld_label(step['weld'], *own_other[step['weld']])} -> "
              f"{int(accessible.sum())}/{len(accessible)} accessible")

    out_path = Path(args.out_dir) / f"showcase_data_{num_samples}.npz"
    np.savez_compressed(out_path, **data)
    print(f"Saved '{out_path.name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="One-off export (run where the full weld_env works): replays "
                     "weld_sequence_from_docx.py on Assem2 for 10 and 2000 sample points and saves "
                     "everything showcase.py needs to redraw it with only numpy + pyvista.")
    parser.add_argument("--step", default=str(THIS_DIR / "Assem2.step"))
    parser.add_argument("--docx", default=str(THIS_DIR / "Assem2.docx"))
    parser.add_argument("--near_tol", type=float, default=0.5)
    parser.add_argument("--proximity_tol", type=float, default=2.0)
    parser.add_argument("--parallel_tol_deg", type=float, default=20.0)
    parser.add_argument("--out_dir", default=str(THIS_DIR / "showcase"))
    args = parser.parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)

    specs = parse_weld_lines_from_docx(args.docx)
    shape = load_step(args.step)
    solids, faces, component_of_face = identify_components(shape)
    n_components = len(solids)
    face_groups = find_cylinder_face_groups(faces, component_of_face=component_of_face)

    tagged_mesh = build_face_tagged_mesh(args.step, faces)
    tagged_mesh.cell_data["ComponentID"] = \
        np.array(component_of_face, dtype=np.int32)[tagged_mesh.cell_data["FaceID"]]
    candidate_edges, own_info = collect_candidate_edges_assembly(shape, faces, face_groups, component_of_face)

    # Shared by both cases: the mesh (with face + component tags) and the numbered candidate edges.
    mesh_out = pv.PolyData(tagged_mesh.points, tagged_mesh.faces)
    mesh_out.cell_data["FaceID"] = np.asarray(tagged_mesh.cell_data["FaceID"])
    mesh_out.cell_data["ComponentID"] = np.asarray(tagged_mesh.cell_data["ComponentID"])
    mesh_out.save(Path(args.out_dir) / "assem2_mesh.vtp")
    np.savez_compressed(
        Path(args.out_dir) / "candidate_edges.npz",
        polylines=np.stack([_sample_edge_polyline(e, EDGE_POLY_POINTS) for e in candidate_edges]),
        label_points=np.stack([_sample_edge_polyline(e, 20)[10] for e in candidate_edges]),
        n_components=np.int32(n_components))

    for num_samples in SAMPLE_CASES:
        export_case(num_samples, tagged_mesh, faces, face_groups, component_of_face, n_components,
                    candidate_edges, own_info, specs, args)

    print(f"\nDone. Copy the '{args.out_dir}' folder to the other machine and run showcase.py there.")


if __name__ == "__main__":
    main()
