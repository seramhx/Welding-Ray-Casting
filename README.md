# 3D Weld Line Torch Accessibility Analysis Engine

An automated Python pipeline to detect concave weld seams in 3D CAD models (`.stp` / `.step`) and evaluate torch accessibility using **OpenCASCADE**, **Gmsh**, and **NVIDIA Warp GPU-accelerated ray casting**.

---

## 📌 What This Script Does

The pipeline performs the following steps:

1. **Concave Seam Detection**
   Parses CAD geometry and B-Rep topology to identify concave joint edges suitable for welding.

2. **High-Density Gmsh Meshing**
   Converts CAD shapes into fine triangular surface meshes for accurate spatial geometry analysis.

3. **Mesh-Level Surface Normal Extraction**
   Uses a `KDTree` proximity search to sample discrete mesh normals (`n₁`, `n₂`) along both curved and flat seams.

4. **Local Joint Coordinate Frame Construction**
   Evaluates local orthonormal frames (`u`, `v`, `t`) at each point along the seam to calculate valid work angles (`φ`) inside the open joint wedge and push/drag angles (`β`).

5. **GPU Ray-Casting Collision Check**
   Casts thousands of torch-clearance rays simultaneously on the GPU using **NVIDIA Warp** to check for collisions against the surrounding part geometry.

6. **3D Interactive Visualization**
   Renders three sequential PyVista interactive windows showing the detected weld seams, local normal vectors, and clear vs. obstructed torch-access paths.

---

## 🛠️ Environment Setup

The project uses a dedicated Conda environment with **Python 3.10**, which is required for compatibility with OpenCASCADE and NVIDIA Warp.

### 1. Create and activate the Conda environment

```bash
conda create -n weld_env python=3.10 -c conda-forge -y
conda activate weld_env
```

### 2. Install CAD, meshing, scientific computing, and visualization dependencies

```bash
conda install -c conda-forge pythonocc-core pyvista trimesh numpy gmsh scipy -y
```

### 3. Install NVIDIA Warp

```bash
pip install warp-lang
```

### Hardware Requirements

An **NVIDIA CUDA-capable GPU** is recommended for fast GPU ray casting.

NVIDIA Warp can also fall back to CPU execution when a compatible GPU is unavailable, although performance may be significantly slower.

---

## 🚀 How to Run

Navigate to the directory containing the script and run the pipeline with a STEP file.

### Basic Run

The following command randomly selects one detected weld line:

```bash
python weldfinal.py --step Part1.stp
```

### Test a Specific Weld Line

To test a specific weld line and customize the torch parameters:

```bash
python weldfinal.py \
    --step Part1.stp \
    --edge_index 0 \
    --tool_radius 1.0 \
    --near_tol 2.0
```

---

## ⚙️ Command-Line Arguments

| Argument          | Type    |      Default | Description                                                                                  |
| ----------------- | ------- | -----------: | -------------------------------------------------------------------------------------------- |
| `--step`          | `str`   | **Required** | Path to the input STEP CAD file (`.stp` or `.step`).                                         |
| `--edge_index`    | `int`   |         `-1` | Index of the concave weld line to test (0-based). If `-1`, a weld line is selected randomly. |
| `--tool_radius`   | `float` |        `5.0` | Physical torch/nozzle clearance radius in **mm**.                                            |
| `--near_tol`      | `float` |        `1.0` | Near-field start offset in **mm**. Allows minor initial corner clipping.                     |
| `--n_dirs`        | `int`   |        `120` | Target number of candidate torch-strategy angles generated per point.                        |
| `--max_push_drag` | `float` |       `60.0` | Maximum push/drag angle sweep along the seam trajectory in **degrees**.                      |

---

## 📊 Visual Output Sequence

When the script runs successfully, it opens **three consecutive interactive 3D PyVista viewer windows**.

### 1. Vis 1 — All Concave Weld Lines

The first visualization displays the complete meshed CAD model and all detected concave weld lines.

* The **fine Gmsh triangular mesh** is displayed for spatial reference.
* **All detected weld lines** are highlighted in **yellow**.
* The **selected weld line** is highlighted in **magenta**.

This view provides an overview of the detected welding locations and identifies which seam is being analyzed.

---

### 2. Vis 2 — Selected Line Local Frames & Mesh Normals

The second visualization focuses on the selected weld line and its local geometric coordinate frames.

The visualization shows:

* Adjacent face meshes surrounding the weld seam.
* Local joint coordinate frames.
* Surface normals sampled from the mesh.
* Seam tangent directions.

#### Vector Colors

| Color     | Vector | Description                                |
| --------- | ------ | ------------------------------------------ |
| 🔴 Red    | `u`    | Local joint bisector direction             |
| 🟢 Green  | `t`    | Local seam tangent direction               |
| 🔵 Cyan   | `n₁`   | Surface normal of the first adjacent face  |
| 🟠 Orange | `n₂`   | Surface normal of the second adjacent face |

These vectors are used to establish the local welding coordinate system and determine valid torch orientations within the joint geometry.

---

### 3. Vis 3 — Torch Accessibility Ray Casting

The third visualization displays the results of the torch accessibility analysis.

Thousands of candidate torch-clearance rays are evaluated against the surrounding CAD geometry.

#### Ray Colors

* 🟢 **Green Rays** — Unobstructed torch paths that successfully clear the part and reach the bounding-box exit.
* 🔴 **Red Rays** — Obstructed torch paths that collide with the part.

Obstructed rays are truncated precisely at their detected surface collision point.

This visualization provides a direct representation of which torch orientations are physically accessible at each sampled position along the weld seam.

---

## 🔬 Analysis Workflow

At a high level, the complete processing pipeline is:

```text
STEP / STP CAD Model
        │
        ▼
OpenCASCADE B-Rep Analysis
        │
        ▼
Concave Weld Seam Detection
        │
        ▼
Gmsh Surface Meshing
        │
        ▼
Mesh Normal Extraction
        │
        ▼
Local Seam Coordinate Frames
        │
        ▼
Candidate Torch Orientations
        │
        ▼
NVIDIA Warp Ray Casting
        │
        ▼
Collision / Accessibility Analysis
        │
        ▼
PyVista 3D Visualization
```

---

## 📐 Local Coordinate System

For each sampled point along the weld seam, a local coordinate frame is constructed.

The primary vectors are:

* **`t`** — Tangent to the weld seam.
* **`n₁`** — Surface normal of the first adjacent face.
* **`n₂`** — Surface normal of the second adjacent face.
* **`u`** — Local joint bisector.
* **`v`** — Remaining orthogonal direction completing the local coordinate frame.

These vectors provide the geometric basis for evaluating torch orientation and welding angles.

The resulting coordinate system is used to evaluate:

* Work angle (`φ`)
* Push/drag angle (`β`)
* Candidate torch directions
* Torch clearance
* Collision-free accessibility

---

## 🎯 Torch Accessibility Analysis

The accessibility analysis considers the physical dimensions of the torch/nozzle rather than treating the torch as an infinitely small point.

The `--tool_radius` parameter defines the effective clearance radius around the torch.

For each candidate torch orientation, the system performs ray-casting against the surrounding mesh to determine whether the torch can travel through the local joint region without intersecting the part.

A path is classified as:

* **Accessible** — The candidate direction clears the surrounding geometry.
* **Obstructed** — The candidate direction intersects the surrounding geometry.

The `--near_tol` parameter provides a small near-field tolerance so that minor initial clipping near the joint corner does not immediately invalidate an otherwise viable torch path.

---

## ✅ Example Commands

### Analyze a randomly selected weld line

```bash
python weldfinal.py --step models/Part3.stp
```

### Analyze weld line 0

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --edge_index 0
```

### Use a larger torch radius

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --edge_index 0 \
    --tool_radius 8.0
```

### Increase the near-field tolerance

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --edge_index 0 \
    --near_tol 2.0
```

### Generate more candidate torch directions

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --n_dirs 240
```

### Increase the allowable push/drag sweep

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --max_push_drag 90.0
```

### Full custom example

```bash
python weldfinal.py \
    --step models/Part3.stp \
    --edge_index 0 \
    --tool_radius 6.0 \
    --near_tol 1.5 \
    --n_dirs 240 \
    --max_push_drag 90.0
```
---
