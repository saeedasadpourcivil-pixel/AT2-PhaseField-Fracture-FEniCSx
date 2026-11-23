# AT2-PhaseField-Fracture-FEniCSx
Finite element implementation of the AT2 phase-field fracture model using FEniCSx:
# AT2 Phase-Field Fracture of a Single-Edge-Cracked Plate

This repository uses FEniCSx to simulate the Miehe-type single-edge tension benchmark. The two-dimensional model uses the AT2 phase-field formulation under plane-strain conditions. The initial crack is represented by two coincident faces with separate mesh nodes along the crack and a shared node at the crack tip.

## Model setup

| Parameter | Value |
|---|---:|
| Plate dimensions | 1 × 1 mm |
| Initial crack length | 0.5 mm |
| Young’s modulus | 210 GPa |
| Poisson’s ratio | 0.30 |
| Critical fracture energy | 2.7 N/mm |
| Phase-field length scale | 0.0075 mm |

The bottom edge is clamped. A prescribed vertical displacement is applied to the top edge, which is free to move horizontally.

## Run the simulation

Run these commands from the repository root in a Python environment with Gmsh and FEniCSx installed:

```bash
python mesh_zero_width_crack.py
python main.py
```

The first command generates `metal_mesh_zero_width.msh`. The second runs the simulation and writes the outputs to `results_zero_width/`.



