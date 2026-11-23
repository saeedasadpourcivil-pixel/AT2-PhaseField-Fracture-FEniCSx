"""
mesh_zero_width_crack.py
Experimental zero-width edge-crack mesh for Miehe-type SEN tension.

Two coincident but topologically DISTINCT crack faces run from (0,H/2)
to (a,H/2). They share the crack-tip vertex and the intact interface
from x=a to x=L. All coordinates are in mm in the .msh output, as
expected by the project's main.py.

Run:
    python mesh_zero_width_crack.py

Output:
    metal_mesh_zero_width.msh

IMPORTANT: Do not overwrite the baseline mesh before backing it up.
The geometry alone does not establish equivalence to the Miehe paper.
"""
from pathlib import Path
import numpy as np
import gmsh
from parameters import geometry

MM = 1e3


def create_mesh():
    L = float(geometry.L) * MM
    H = float(geometry.H) * MM
    a = float(geometry.notch_length) * MM
    h_tip = float(geometry.h_tip) * MM
    h_band = float(geometry.h_target) * MM
    h_far = float(geometry.h_far) * MM
    band = float(geometry.band_half_width) * MM
    yc = H / 2.0
    if not (0 < a < L and 0 < h_tip <= h_band <= h_far and band > 0):
        raise ValueError("Invalid geometry/mesh settings.")
    out = Path(__file__).resolve().parent / "metal_mesh_zero_width.msh"

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("sen_zero_width_crack")
        geo = gmsh.model.geo

        # Upper and lower domains have independent points at the crack mouth.
        # The tip and right midpoint are SHARED to preserve intact material.
        bl = geo.addPoint(0, 0, 0)
        br = geo.addPoint(L, 0, 0)
        tr = geo.addPoint(L, H, 0)
        tl = geo.addPoint(0, H, 0)
        ml_upper = geo.addPoint(0, yc, 0)
        ml_lower = geo.addPoint(0, yc, 0)
        tip = geo.addPoint(a, yc, 0)
        rm = geo.addPoint(L, yc, 0)

        bottom = geo.addLine(bl, br)
        right_lower = geo.addLine(br, rm)
        right_upper = geo.addLine(rm, tr)
        top = geo.addLine(tr, tl)
        left_upper = geo.addLine(tl, ml_upper)
        left_lower = geo.addLine(ml_lower, bl)

        # Distinct geometrical curves, identical spatial coordinates.
        crack_upper = geo.addLine(ml_upper, tip)
        crack_lower = geo.addLine(tip, ml_lower)
        intact = geo.addLine(tip, rm)

        # Upper loop: tip -> right midpoint -> top-right -> top-left
        #             -> upper mouth -> tip
        upper_loop = geo.addCurveLoop([
            intact, right_upper, top, left_upper, crack_upper
        ])
        # Lower loop: bottom-left -> bottom-right -> right midpoint
        #             -> tip -> lower mouth -> bottom-left
        lower_loop = geo.addCurveLoop([
            bottom, right_lower, -intact, crack_lower, left_lower
        ])
        upper = geo.addPlaneSurface([upper_loop])
        lower = geo.addPlaneSurface([lower_loop])
        geo.synchronize()

        pg = gmsh.model.addPhysicalGroup(2, [upper, lower], 1)
        gmsh.model.setPhysicalName(2, pg, "metal")

        boundaries = [
            (11, "bottom", [bottom]),
            (12, "right", [right_lower, right_upper]),
            (13, "top", [top]),
            (14, "left_upper", [left_upper]),
            (15, "crack_upper", [crack_upper]),
            (16, "crack_tip_interface", [intact]),
            (17, "crack_lower", [crack_lower]),
            (18, "left_lower", [left_lower]),
        ]
        for tag, name, curves in boundaries:
            phys = gmsh.model.addPhysicalGroup(1, curves, tag)
            gmsh.model.setPhysicalName(1, phys, name)

        # Refined horizontal strip starting slightly before the crack tip.
        strip = gmsh.model.mesh.field.add("Box")
        for name, value in {
            "VIn": h_band, "VOut": h_far,
            "XMin": max(0.0, a - band), "XMax": L,
            "YMin": yc - band, "YMax": yc + band,
            "ZMin": -1.0, "ZMax": 1.0,
        }.items():
            gmsh.model.mesh.field.setNumber(strip, name, value)

        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "PointsList", [tip])
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_tip)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 2*h_tip)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", band)

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [strip, threshold])
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_tip)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)
        gmsh.model.mesh.generate(2)

        # Check that crack-face mesh nodes are independent at the same x.
        upper_nodes, upper_coords, _ = gmsh.model.mesh.getNodes(1, crack_upper)
        lower_nodes, lower_coords, _ = gmsh.model.mesh.getNodes(1, crack_lower)
        upper_nodes = set(map(int, upper_nodes))
        lower_nodes = set(map(int, lower_nodes))
        if not upper_nodes or not lower_nodes:
            raise RuntimeError("Crack faces have no interior mesh nodes.")
        if upper_nodes & lower_nodes:
            raise RuntimeError("Crack faces share nodes; crack is not mechanically open.")

        # Check that no elements other than linear triangles are present.
        types, element_tags, _ = gmsh.model.mesh.getElements(2)
        ntri = 0
        for element_type, tags in zip(types, element_tags):
            _, dim, order, nn, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dim == 2:
                if order != 1 or nn != 3:
                    raise RuntimeError("Unexpected non-T3 element in mesh.")
                ntri += len(tags)

        gmsh.write(str(out))
        print(f"Saved: {out}")
        print(f"Triangles: {ntri}")
        print(f"Crack-face interior nodes: upper={len(upper_nodes)}, lower={len(lower_nodes)}")
        print(f"Tip target: {h_tip:.6f} mm; band target: {h_band:.6f} mm; far: {h_far:.6f} mm")
        print("Crack is a zero-width geometric cut; d=0 initially is appropriate.")
        print("NOTE: main.py must load metal_mesh_zero_width.msh to use this mesh.")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    create_mesh()
