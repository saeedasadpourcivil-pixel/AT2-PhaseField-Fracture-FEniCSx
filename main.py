"""Run the zero-width-crack geometry with the existing AT2 solver.

First: python mesh.py
Then:  python main_zero_width.py

Requires parameters.py and fracture_solver.py in the same directory.
"""
from pathlib import Path



import numpy as np

import matplotlib



SHOW_FIGURES = False  # True: open windows (blocks until closed)

if not SHOW_FIGURES:

    matplotlib.use("Agg")



import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection

from matplotlib.tri import Triangulation

from mpi4py import MPI

from dolfinx.io import gmsh



from parameters import geometry, material, loading, validate_parameters

from fracture_solver import run_simulation, mesh_extents

import fracture_solver

print("Solver file being used:", fracture_solver.__file__)



MM = 1.0e3            # metres -> millimetres for plots

THICKNESS_MM = 1.0    # plate thickness assumed for the force in kN





def convert_mesh_units(domain):

    """

    mesh.py writes the mesh in millimetres. Convert it to metres here.

    A mesh that is already in metres is kept as it is.

    Returns a short description of what was done.

    """

    x_min, x_max, _, _ = mesh_extents(domain)

    ratio = (x_max - x_min) / float(geometry.L)



    if abs(ratio / MM - 1.0) < 0.05:

        domain.geometry.x[:] *= 1.0 / MM

        return "mm (converted to m)"

    if abs(ratio - 1.0) < 0.05:

        return "m"

    return f"unknown (width/L = {ratio:.4g})"





def load_mesh(mesh_file):

    """Load a Gmsh mesh using the DOLFINx gmsh reader."""

    mesh_file = Path(mesh_file).resolve()



    if not mesh_file.exists():

        raise FileNotFoundError(

            f"Mesh file not found: {mesh_file}\n"

            "Run python mesh.py first."

        )



    mesh_data = gmsh.read_from_msh(

        str(mesh_file),

        MPI.COMM_WORLD,

        rank=0,

        gdim=2,

    )

    domain = mesh_data.mesh

    unit = convert_mesh_units(domain)

    if MPI.COMM_WORLD.rank == 0:

        print(f"Length unit in the mesh file: {unit}")



    if unit.startswith("unknown"):

        x_min, x_max, y_min, y_max = mesh_extents(domain)

        raise RuntimeError(

            "The selected mesh does not match parameters.py "

            f"(mesh size: {x_max - x_min:.6g} x {y_max - y_min:.6g}; "

            f"parameters: {float(geometry.L):.6g} x "

            f"{float(geometry.H):.6g} m). It is probably an old mesh.\n"

            "Regenerate the zero-width mesh and rerun main.py."

        )

    return domain





def get_mesh_polygons(domain):

    """Build polygons directly from the mesh geometry (P1 tri/quad)."""

    from dolfinx.mesh import CellType



    points = domain.geometry.x[:, :2]

    try:

        cells = domain.geometry.dofmaps[0]

    except (AttributeError, IndexError, TypeError):

        cells = domain.geometry.dofmap

    cell_type = domain.topology.cell_type



    if cell_type == CellType.triangle:

        order = [0, 1, 2]

    elif cell_type == CellType.quadrilateral:

        # DOLFINx quads are tensor-product ordered: reorder to a loop.

        order = [0, 1, 3, 2]

    else:

        raise RuntimeError(f"Unsupported mesh cell type: {cell_type}")



    polygons = [points[cell[order]] for cell in cells]



    if not polygons:

        raise RuntimeError("No mesh cells were found for plotting.")



    return polygons





def finish_figure(fig, filename, dpi):

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    print(f"Saved: {filename}")

    if SHOW_FIGURES:

        plt.show()

    plt.close(fig)





def draw_mesh(ax, polygons_mm, L, H, notch_length, notch_half_width,

              title, linewidth):

    """Draw the mesh (all lengths in mm) and highlight the notch."""

    collection = PolyCollection(

        polygons_mm,

        facecolors="white",

        edgecolors="#303030",

        linewidths=linewidth,

        antialiaseds=False,

    )

    ax.add_collection(collection)



    # Zero-width crack: plot as a line, not a filled rectangular notch.
    ax.plot([0.0, notch_length], [H / 2.0, H / 2.0],
            color="red", linewidth=1.2, zorder=5,
            label="Initial zero-width crack")

    ax.set_xlim(0.0, L)

    ax.set_ylim(0.0, H)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x [mm]")

    ax.set_ylabel("y [mm]")

    ax.set_title(title)

    ax.legend(loc="upper right", fontsize=8)





def save_mesh_figures(domain, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    L = float(geometry.L) * MM

    H = float(geometry.H) * MM

    a = float(geometry.notch_length) * MM

    half_w = 0.0  # Not used for the zero-width crack



    polygons_mm = [p * MM for p in get_mesh_polygons(domain)]

    print(f"Number of cells: {len(polygons_mm)}")



    fig, ax = plt.subplots(figsize=(8, 8))

    draw_mesh(ax, polygons_mm, L, H, a, half_w,

              f"Full plate mesh ({len(polygons_mm)} cells)", 0.1)

    fig.tight_layout()

    finish_figure(fig, output_dir / "01_full_mesh.png", 300)



    # Close-up around the notch tip

    fig2, ax2 = plt.subplots(figsize=(8, 7))

    draw_mesh(ax2, polygons_mm, L, H, a, half_w,

              "Mesh near the notch tip", 0.4)



    radius = 6.0 * float(material.ell) * MM

    ax2.set_xlim(a - radius, a + 3.0 * radius)

    ax2.set_ylim(H / 2.0 - 2.0 * radius, H / 2.0 + 2.0 * radius)

    fig2.tight_layout()

    finish_figure(fig2, output_dir / "02_notch_tip_mesh.png", 300)





def plot_damage(d, output_dir):

    """Plot the final damage field (serial runs)."""

    L = float(geometry.L) * MM

    H = float(geometry.H) * MM



    Q = d.function_space

    coords = Q.tabulate_dof_coordinates()[:, :2] * MM

    cells = np.asarray(Q.dofmap.list)



    if cells.shape[1] == 3:

        triangles = cells

    elif cells.shape[1] == 4:

        triangles = np.vstack([cells[:, [0, 1, 3]], cells[:, [0, 3, 2]]])

    else:

        raise RuntimeError("Unsupported cell type for damage plot.")



    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)



    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.tripcolor(

        tri, d.x.array[: len(coords)], shading="gouraud",

        vmin=0.0, vmax=1.0, cmap="jet",

    )

    fig.colorbar(image, ax=ax, label="damage d [-]", shrink=0.85)

    ax.set_xlim(0.0, L)

    ax.set_ylim(0.0, H)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x [mm]")

    ax.set_ylabel("y [mm]")

    ax.set_title("Final damage field")

    fig.tight_layout()

    finish_figure(fig, Path(output_dir) / "03_damage.png", 300)





def read_curve(result, output_dir):

    """Get the force-displacement data from the result or the CSV."""

    if isinstance(result, dict) and len(result.get("force", [])) > 0:

        return (

            np.asarray(result["displacement"], dtype=float),

            np.asarray(result["force"], dtype=float),

        )



    csv_path = Path(output_dir) / "load_results.csv"

    if csv_path.exists():

        data = np.genfromtxt(csv_path, delimiter=",", names=True)

        return (

            np.atleast_1d(data["displacement_m"]).astype(float),

            np.atleast_1d(data["reaction_N_per_m"]).astype(float),

        )



    return None, None





def plot_force_displacement(result, output_dir):

    displacement, force = read_curve(result, output_dir)



    if displacement is None or len(displacement) < 2:

        print("Not enough data to plot the force-displacement curve.")

        return



    # N/m (per unit thickness) -> kN for a plate of THICKNESS_MM

    force_kn = force * (THICKNESS_MM * 1.0e-3) * 1.0e-3

    disp_mm = displacement * MM



    disp_mm = np.concatenate([[0.0], disp_mm])

    force_kn = np.concatenate([[0.0], force_kn])



    i_peak = int(np.argmax(force_kn))

    print(

        f"Peak force: {force_kn[i_peak]:.4f} kN at "

        f"u = {disp_mm[i_peak]:.5f} mm "

        f"(thickness {THICKNESS_MM:g} mm)"

    )



    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(disp_mm, force_kn, "-", linewidth=1.5)

    ax.plot(disp_mm[i_peak], force_kn[i_peak], "ro", markersize=5,

            label=f"peak: {force_kn[i_peak]:.3f} kN at "

                  f"{disp_mm[i_peak]:.4f} mm")

    ax.set_xlabel("Applied displacement [mm]")

    ax.set_ylabel(f"Reaction force [kN] (thickness {THICKNESS_MM:g} mm)")

    ax.set_title("Force-displacement response")

    ax.grid(True, alpha=0.3)

    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    finish_figure(fig, Path(output_dir) / "04_force_displacement.png", 300)





def paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def plot_geometry_paper(output_dir):
    """Schematic, not a claim about the source article's exact CAD details."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    L = float(geometry.L) * MM
    H = float(geometry.H) * MM
    a = float(geometry.notch_length) * MM
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    ax.add_patch(Rectangle((0, 0), L, H, facecolor="#f4f7fa",
                           edgecolor="#253447", lw=1.7))
    ax.plot([0, a], [H/2, H/2], color="#c52d2f", lw=2.6,
            solid_capstyle="round", label="Initial crack")
    ax.plot(a, H/2, "o", color="#c52d2f", ms=4)
    # Clamped lower edge
    for x in np.linspace(0.025*L, 0.975*L, 20):
        ax.plot([x-0.012*L, x+0.012*L], [-0.035*H, -0.005*H],
                color="#64748b", lw=0.9)
    ax.plot([0, L], [0, 0], color="#253447", lw=2.5)
    for x in np.linspace(0.10*L, 0.90*L, 5):
        ax.annotate("", xy=(x, H+0.15*H), xytext=(x, H+0.025*H),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#2463a6"))
    ax.text(L/2, H+0.19*H, "Prescribed vertical displacement",
            ha="center", color="#2463a6", fontsize=10)
    ax.annotate("", xy=(0.5*L, -0.17*H), xytext=(0.0, -0.17*H),
                arrowprops=dict(arrowstyle="<->", color="#253447"))
    ax.text(0.25*L, -0.205*H, "1 mm", ha="center")
    ax.annotate("", xy=(L+0.13*L, H), xytext=(L+0.13*L, 0),
                arrowprops=dict(arrowstyle="<->", color="#253447"))
    ax.text(L+0.16*L, H/2, "1 mm", rotation=90, va="center")
    ax.annotate("Crack tip", xy=(a, H/2), xytext=(0.59*L, 0.39*H),
                arrowprops=dict(arrowstyle="->", color="#c52d2f"),
                color="#c52d2f")
    ax.set_xlim(-0.13*L, 1.38*L)
    ax.set_ylim(-0.25*H, 1.27*H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    finish_figure(fig, Path(output_dir)/"00_geometry_paper.png", 450)
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    ax.add_patch(Rectangle((0, 0), L, H, facecolor="#f4f7fa",
                           edgecolor="#253447", lw=1.5))
    ax.plot([0, a], [H/2, H/2], color="#c52d2f", lw=2.4)
    ax.axhspan(H/2-0.03*H, H/2+0.03*H, xmin=a/L, xmax=1,
               color="#e8b84b", alpha=0.3, label="Refined crack path")
    ax.set_xlim(0, L)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Geometry and refined propagation band")
    fig.tight_layout()
    finish_figure(fig, Path(output_dir)/"00_refinement_region.png", 450)


def save_damage_snapshot(domain, d, displacement, reaction, step, output_dir):
    """Save crack-growth panels during the solve; serial execution only."""
    Q = d.function_space
    coords = Q.tabulate_dof_coordinates()[:, :2] * MM
    cells = np.asarray(Q.dofmap.list)
    if cells.shape[1] == 3:
        triangles = cells
    elif cells.shape[1] == 4:
        triangles = np.vstack((cells[:, [0,1,3]], cells[:, [0,3,2]]))
    else:
        raise RuntimeError("Damage plot requires P1 triangles/quads.")
    triang = Triangulation(coords[:,0], coords[:,1], triangles)
    values = np.asarray(d.x.array[:len(coords)])
    folder = Path(output_dir)/"crack_growth"
    folder.mkdir(parents=True, exist_ok=True)
    u_mm = displacement * MM
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    im = ax.tripcolor(triang, values, shading="gouraud",
                      cmap="jet", vmin=0, vmax=1, rasterized=True)
    ax.plot([0, float(geometry.notch_length)*MM],
            [float(geometry.H)*MM/2]*2, ls="--", color="black",
            lw=1, label="Initial crack")
    ax.set_xlim(0, float(geometry.L)*MM)
    ax.set_ylim(0, float(geometry.H)*MM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Crack evolution | u = {u_mm:.4f} mm | "
                 f"F = {reaction*1e-6:.3f} kN")
    fig.colorbar(im, ax=ax, label="Phase-field damage d", shrink=0.83)
    fig.tight_layout()
    filename = folder/f"damage_u_{u_mm:.4f}mm_step_{step:05d}.png"
    finish_figure(fig, filename, 350)




def save_initial_damage(domain, output_dir):
    """Undamaged bulk (d=0) at u=0; initial crack is geometric."""
    coords = domain.geometry.x[:, :2] * MM
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)
    cells = np.array([conn.links(i) for i in range(
        domain.topology.index_map(tdim).size_local)], dtype=np.int32)
    if cells.shape[1] != 3:
        raise RuntimeError("Initial damage figure requires triangular mesh.")
    triang = Triangulation(coords[:, 0], coords[:, 1], cells)
    folder = Path(output_dir) / "crack_growth"
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    im = ax.tripcolor(triang, np.zeros(len(coords)), shading="flat",
                      cmap="jet", vmin=0, vmax=1, rasterized=True)
    a = float(geometry.notch_length) * MM
    y = float(geometry.H) * MM / 2
    ax.plot([0, a], [y, y], color="black", lw=2.3,
            solid_capstyle="round", label="Geometric initial crack")
    ax.set_xlim(0, float(geometry.L) * MM)
    ax.set_ylim(0, float(geometry.H) * MM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Initial damage field | u = 0 mm | d = 0 in bulk")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="Phase-field damage d", shrink=0.82)
    fig.tight_layout()
    finish_figure(fig, folder / "00_initial_damage.png", 350)


def main():

    comm = MPI.COMM_WORLD

    rank = comm.rank



    validate_parameters()
    paper_style()

    project_dir = Path(__file__).resolve().parent

    mesh_file = project_dir / "metal_mesh_zero_width.msh"

    output_dir = project_dir / "results_zero_width"



    domain = load_mesh(mesh_file)



    L = float(geometry.L)

    H = float(geometry.H)



    x_min, x_max, y_min, y_max = mesh_extents(domain)

    if rank == 0:

        print("Mesh loaded successfully.")

        print(

            f"Mesh extents : x = [{x_min * MM:.5g}, {x_max * MM:.5g}] mm, "

            f"y = [{y_min * MM:.5g}, {y_max * MM:.5g}] mm"

        )

        print(f"parameters.py: L = {L * MM:.5g} mm, H = {H * MM:.5g} mm")

        if (

            abs(x_max - x_min - L) > 1.0e-3 * L

            or abs(y_max - y_min - H) > 1.0e-3 * H

        ):

            print(

                "WARNING: the mesh does not match parameters.py. "

                "Run python mesh.py again."

            )



    if comm.size == 1:

        plot_geometry_paper(output_dir)
        save_mesh_figures(domain, output_dir)
        save_initial_damage(domain, output_dir)

    comm.barrier()

    import sys
    if "--initial-only" in sys.argv:
        if rank == 0:
            print("Initial figures saved; simulation skipped.")
        return

    try:
        result = run_simulation(

        domain=domain,

        E=material.E,

        nu=material.nu,

        Gc=material.Gc,

        ell=material.ell,

        kappa=material.kappa,

        energy_split=material.energy_split,

        length=L,

        height=H,

        max_displacement=loading.max_displacement,

        increment_stage1=1.0e-8,  # 1e-5 mm in metres

        steps_stage1=500,

        increment_stage2=1.0e-9,  # 1e-6 mm in metres

        output_dir=str(output_dir),
        output_every=5,
        max_stagger_iterations=100,
        stagger_tolerance=3.0e-3,
        paper_fixed_steps=True,
        staggered_scheme="one_pass",
        verbose_snes=True,
        min_increment_ratio=1.0/1024.0,
        snapshot_targets_mm=(0.0050, 0.0052, 0.0053, 0.0055,
                             0.0057, 0.0059, 0.0061, 0.0063),
        snapshot_callback=save_damage_snapshot if comm.size == 1 else None,
        )
    except RuntimeError as exc:
        if rank == 0:
            print(f"Simulation stopped without accepting an unconverged state: {exc}")
            print("Previously accepted steps are saved in load_results.csv / fields.xdmf.")
        if comm.size == 1:
            plot_force_displacement(None, output_dir)
        raise

    if comm.size == 1:

        plot_damage(result["d"], output_dir)

        plot_force_displacement(result, output_dir)

    elif rank == 0:

        print("Plots are skipped in parallel. Use fields.xdmf in ParaView.")





if __name__ == "__main__":

    main()