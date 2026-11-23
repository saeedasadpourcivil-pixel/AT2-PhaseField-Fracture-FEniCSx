"""

fracture_solver.py

------------------

AT2 phase-field fracture solver (DOLFINx).



- 2D plane strain, small strains, no plasticity

- Energy split: spectral (Miehe et al. 2010, default) or

  volumetric-deviatoric; 3D constitutive law with eps_zz = 0

- Staggered scheme: displacement (Newton) -> history -> damage (linear)

- History field on DG0 (psi_plus is cell-wise constant for P1 displacements)

- Notch is a real gap in the mesh; damage starts from zero everywhere

- Displacement-controlled tension in y: bottom edge clamped

  (ux = uy = 0), top edge uy = applied displacement, ux free



Reaction force is per unit thickness [N/m] in 2D.

"""



from pathlib import Path

import csv



import numpy as np

import ufl

from mpi4py import MPI

from petsc4py import PETSc

from dolfinx import fem, io

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

from dolfinx.mesh import locate_entities_boundary, meshtags





class StepFailed(RuntimeError):

    """A load step failed and may be retried with a smaller increment."""





# ============================================================

# Spaces and helpers

# ============================================================





def create_spaces(domain):

    if domain.geometry.dim != 2:

        raise ValueError("This solver requires a 2D mesh.")



    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # displacement

    Q = fem.functionspace(domain, ("Lagrange", 1))        # damage

    W = fem.functionspace(domain, ("DG", 0))              # history

    return V, Q, W





def interpolation_points(space):

    """Work with DOLFINx versions where this is a property or a method."""

    points = space.element.interpolation_points

    return points() if callable(points) else points





def global_l2_norm(values, n_owned, comm):

    local_sq = float(np.dot(values[:n_owned], values[:n_owned]))

    return float(np.sqrt(comm.allreduce(local_sq, op=MPI.SUM)))





def global_max_abs(values, n_owned, comm):

    local = (

        float(np.max(np.abs(values[:n_owned]))) if n_owned > 0 else 0.0

    )

    return comm.allreduce(local, op=MPI.MAX)





# ============================================================

# Constitutive model

# ============================================================





def elastic_parameters(E, nu):

    if E <= 0.0:

        raise ValueError("Young's modulus E must be positive.")

    if not -1.0 < nu < 0.5:

        raise ValueError("Poisson ratio nu must lie between -1 and 0.5.")



    mu = E / (2.0 * (1.0 + nu))

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    K = lam + 2.0 * mu / 3.0

    return lam, mu, K





def strain(u):

    return ufl.sym(ufl.grad(u))





def strain_plane_strain_3d(u):

    """Embed the 2D strain into 3D with eps_zz = 0 (plane strain)."""

    e = strain(u)

    return ufl.as_tensor([

        [e[0, 0], e[0, 1], 0.0],

        [e[1, 0], e[1, 1], 0.0],

        [0.0,     0.0,     0.0],

    ])





def _pos(x):

    """Positive part <x>_+ of a UFL scalar."""

    return 0.5 * (x + abs(x))





def spectral_split_plane_strain(u, E, nu, delta=1.0e-12):

    """

    Spectral (principal-strain) split of Miehe et al. (2010) for plane

    strain (eps_zz = 0):



        psi_plus  = lam/2 <tr eps>_+^2 + mu tr(eps_plus^2)

        sigma_plus  = lam <tr eps>_+ I + 2 mu eps_plus

        sigma_minus = lam <tr eps>_- I + 2 mu eps_minus

        eps_plus = sum_a <eps_a>_+ n_a (x) n_a



    The in-plane eigenvalues are written in closed form,

        eps_1,2 = tr/2 +- D,   D = sqrt(Q^2 + R^2),

    with Q = (eps_xx - eps_yy)/2 and R = eps_xy, and the eigenprojectors

    as P_1,2 = (I +- dev/D)/2. `delta` only regularises D at isotropic

    strain. The out-of-plane eigenvalue is 0 and does not contribute.



    Returns psi_plus, sigma_plus, sigma_minus (in-plane 2x2 stresses).

    """

    lam, mu, _ = elastic_parameters(E, nu)



    e = strain(u)

    I2 = ufl.Identity(2)



    tr = e[0, 0] + e[1, 1]

    dev = e - 0.5 * tr * I2



    Q = 0.5 * (e[0, 0] - e[1, 1])

    R = e[0, 1]

    D = ufl.sqrt(Q**2 + R**2 + delta**2)



    p1 = _pos(0.5 * tr + D)

    p2 = _pos(0.5 * tr - D)



    eps_plus = 0.5 * (p1 + p2) * I2 + 0.5 * (p1 - p2) * dev / D

    eps_minus = e - eps_plus



    tr_pos = _pos(tr)

    tr_neg = tr - tr_pos



    psi_plus = 0.5 * lam * tr_pos**2 + mu * (p1**2 + p2**2)

    sigma_plus = lam * tr_pos * I2 + 2.0 * mu * eps_plus

    sigma_minus = lam * tr_neg * I2 + 2.0 * mu * eps_minus



    return psi_plus, sigma_plus, sigma_minus





def tensile_energy_and_stress(

    u, d, E, nu, kappa=1.0e-8, split="spectral"

):

    """

    Driving energy psi_plus and degraded stress for the chosen split.



    split = "spectral"

        Miehe et al. (2010): sigma = g(d) sigma_plus + sigma_minus.

    split = "volumetric_deviatoric"

        psi_plus  = K/2 <tr eps>_+^2 + mu |dev eps|^2

        psi_minus = K/2 <tr eps>_-^2

        (the deviatoric part is degraded in tension and compression).



    g(d) = (1 - d)^2 + kappa. Returns psi_plus (UFL scalar) and the

    in-plane 2x2 stress.

    """

    g = (1.0 - d) ** 2 + kappa



    if split == "spectral":

        psi_plus, sigma_plus, sigma_minus = spectral_split_plane_strain(

            u, E, nu

        )

        return psi_plus, g * sigma_plus + sigma_minus



    if split != "volumetric_deviatoric":

        raise ValueError(

            "split must be 'spectral' or 'volumetric_deviatoric'."

        )



    _, mu, K = elastic_parameters(E, nu)



    eps = strain_plane_strain_3d(u)

    I3 = ufl.Identity(3)



    tr_eps = ufl.tr(eps)

    dev = eps - (tr_eps / 3.0) * I3



    tr_pos = ufl.max_value(tr_eps, 0.0)

    tr_neg = ufl.min_value(tr_eps, 0.0)



    psi_plus = 0.5 * K * tr_pos**2 + mu * ufl.inner(dev, dev)



    sigma_plus = K * tr_pos * I3 + 2.0 * mu * dev

    sigma_minus = K * tr_neg * I3

    sigma3 = g * sigma_plus + sigma_minus



    sigma = ufl.as_tensor([

        [sigma3[0, 0], sigma3[0, 1]],

        [sigma3[1, 0], sigma3[1, 1]],

    ])



    return psi_plus, sigma





# ============================================================

# Boundary conditions and measures

# ============================================================





def mesh_extents(domain):

    """Bounding box of the mesh: (x_min, x_max, y_min, y_max) [m]."""

    comm = domain.comm

    x = domain.geometry.x



    if len(x) > 0:

        lo = x[:, :2].min(axis=0)

        hi = x[:, :2].max(axis=0)

    else:

        lo = np.array([np.inf, np.inf])

        hi = np.array([-np.inf, -np.inf])



    x_min = comm.allreduce(float(lo[0]), op=MPI.MIN)

    y_min = comm.allreduce(float(lo[1]), op=MPI.MIN)

    x_max = comm.allreduce(float(hi[0]), op=MPI.MAX)

    y_max = comm.allreduce(float(hi[1]), op=MPI.MAX)

    return x_min, x_max, y_min, y_max





def create_boundary_conditions(domain, V, length, height, top_value):

    """

    bottom edge: ux = 0 and uy = 0 (fully clamped)

    top edge:    uy = top_value (fem.Constant, updated every load step),

                 ux free



    The bottom and top edges are taken from the mesh itself (lowest and

    highest y). `length` and `height` are only used to detect a mesh that

    does not match parameters.py.

    """

    comm = domain.comm

    tdim = domain.topology.dim

    fdim = tdim - 1



    x_min, x_max, y_min, y_max = mesh_extents(domain)



    mismatch = (

        abs(x_max - x_min - length) > 1.0e-3 * length

        or abs(y_max - y_min - height) > 1.0e-3 * height

    )

    if mismatch:

        mm = 1.0e3

        raise RuntimeError(

            "The mesh does not match parameters.py.\n"

            f"  mesh      : x = [{x_min * mm:.5g}, {x_max * mm:.5g}] mm, "

            f"y = [{y_min * mm:.5g}, {y_max * mm:.5g}] mm\n"

            f"  parameters: L = {length * mm:.5g} mm, "

            f"H = {height * mm:.5g} mm\n"

            "Run python mesh.py again after changing parameters.py."

        )



    tol = 1.0e-6 * max(x_max - x_min, y_max - y_min)



    def bottom(x):

        return np.abs(x[1] - y_min) < tol



    def top(x):

        return np.abs(x[1] - y_max) < tol



    domain.topology.create_connectivity(fdim, tdim)



    bottom_facets = locate_entities_boundary(domain, fdim, bottom)

    top_facets = locate_entities_boundary(domain, fdim, top)



    dofs_bottom_x = fem.locate_dofs_topological(

        V.sub(0), fdim, bottom_facets

    )

    dofs_bottom_y = fem.locate_dofs_topological(

        V.sub(1), fdim, bottom_facets

    )

    dofs_top_y = fem.locate_dofs_topological(

        V.sub(1), fdim, top_facets

    )



    for name, dofs in (

        ("bottom edge (ux)", dofs_bottom_x),

        ("bottom edge (uy)", dofs_bottom_y),

        ("top edge (uy)", dofs_top_y),

    ):

        if comm.allreduce(len(dofs), op=MPI.SUM) == 0:

            raise RuntimeError(

                f"No DOFs found on {name}. Mesh extents: "

                f"x = [{x_min:.6e}, {x_max:.6e}], "

                f"y = [{y_min:.6e}, {y_max:.6e}] m."

            )



    zero = fem.Constant(domain, PETSc.ScalarType(0.0))



    bcs = [

        fem.dirichletbc(zero, dofs_bottom_x, V.sub(0)),

        fem.dirichletbc(zero, dofs_bottom_y, V.sub(1)),

        fem.dirichletbc(top_value, dofs_top_y, V.sub(1)),

    ]

    return bcs, top_facets





def create_top_measure(domain, top_facets):

    """Measure for integrating the reaction along the top edge."""

    fdim = domain.topology.dim - 1

    facets = np.sort(np.asarray(top_facets, dtype=np.int32))

    values = np.ones(len(facets), dtype=np.int32)

    tags = meshtags(domain, fdim, facets, values)



    return ufl.Measure(

        "ds", domain=domain, subdomain_data=tags, subdomain_id=1

    )





# ============================================================

# Main driver

# ============================================================





def run_simulation(

    domain,

    E,

    nu,

    Gc,

    ell,

    length,

    height,

    max_displacement,

    increment_stage1,

    steps_stage1,

    increment_stage2,

    kappa=1.0e-8,

    energy_split="spectral",

    output_dir="results",

    output_every=10,

    max_stagger_iterations=100,

    min_increment_ratio=1.0 / 1024.0,
    paper_fixed_steps=False,
    staggered_scheme="staggered",

    snapshot_targets_mm=(),

    snapshot_callback=None,

    stagger_tolerance=3.0e-3,


    stop_force_fraction=0.01,

    verbose_snes=False,

):

    if staggered_scheme not in ("staggered", "one_pass"):
        raise ValueError("staggered_scheme must be staggered or one_pass")

    if not (0.0 < min_increment_ratio <= 1.0):
        raise ValueError("min_increment_ratio must be in (0, 1].")

    if snapshot_targets_mm is None:
        snapshot_targets_mm = ()

    if E <= 0.0 or Gc <= 0.0 or ell <= 0.0:

        raise ValueError("E, Gc and ell must be positive.")

    if not 0.0 < kappa < 1.0:

        raise ValueError("kappa must satisfy 0 < kappa < 1.")

    if energy_split not in ("spectral", "volumetric_deviatoric"):

        raise ValueError(

            "energy_split must be 'spectral' or 'volumetric_deviatoric'."

        )

    if length <= 0.0 or height <= 0.0:

        raise ValueError("Specimen dimensions must be positive.")

    if (

        max_displacement <= 0.0

        or increment_stage1 <= 0.0

        or increment_stage2 <= 0.0

        or steps_stage1 < 1

    ):

        raise ValueError("Invalid loading parameters.")



    comm = domain.comm

    rank = comm.rank



    V, Q, W = create_spaces(domain)



    u = fem.Function(V, name="displacement")

    d = fem.Function(Q, name="damage")

    d_old = fem.Function(Q, name="damage_previous_step")

    d_trial = fem.Function(Q, name="damage_trial")

    H = fem.Function(W, name="history")

    psi_cell = fem.Function(W, name="psi_plus")



    dx = ufl.Measure("dx", domain=domain)



    # ---------------- boundary conditions ----------------

    top_value = fem.Constant(domain, PETSc.ScalarType(0.0))

    bcs, top_facets = create_boundary_conditions(

        domain, V, length, height, top_value

    )

    top_ds = create_top_measure(domain, top_facets)



    # ---------------- mechanical problem (built once) ----------------

    v = ufl.TestFunction(V)

    du = ufl.TrialFunction(V)



    psi_plus, sigma = tensile_energy_and_stress(

        u, d, E, nu, kappa, split=energy_split

    )



    F = ufl.inner(sigma, strain(v)) * dx

    J = ufl.derivative(F, u, du)



    snes_options = {

        "snes_type": "newtonls",

        "snes_linesearch_type": "bt",

        "snes_rtol": 1.0e-8,

        "snes_atol": 1.0e-3,

        "snes_max_it": 100,

        "ksp_type": "preonly",

        "pc_type": "lu",

    }

    if verbose_snes:

        snes_options["snes_monitor"] = None



    mech_problem = NonlinearProblem(

        F,

        u,

        bcs=bcs,

        J=J,

        petsc_options_prefix="metal_mech_",

        petsc_options=snes_options,

    )



    # ---------------- damage problem (linear, built once) ----------------

    q = ufl.TestFunction(Q)

    dd = ufl.TrialFunction(Q)



    a_damage = (

        (Gc / ell + 2.0 * H) * dd * q

        + Gc * ell * ufl.inner(ufl.grad(dd), ufl.grad(q))

    ) * dx

    L_damage = 2.0 * H * q * dx



    damage_problem = LinearProblem(

        a_damage,

        L_damage,

        u=d_trial,

        bcs=[],

        petsc_options_prefix="metal_damage_",

        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},

    )



    psi_expression = fem.Expression(

        psi_plus, interpolation_points(W)

    )

    reaction_form = fem.form(sigma[1, 1] * top_ds)



    # ---------------- output ----------------

    output_path = Path(output_dir)

    if rank == 0:

        output_path.mkdir(parents=True, exist_ok=True)

    comm.barrier()



    csv_path = output_path / "load_results.csv"

    if rank == 0:

        with csv_path.open("w", newline="") as f:

            csv.writer(f).writerow([

                "step", "displacement_m", "reaction_N_per_m",

                "max_damage", "stagger_iterations", "converged",

                "rel_u_change", "max_d_change",

            ])



    n_u_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    n_d_owned = Q.dofmap.index_map.size_local



    # Adaptive load stepping: the nominal increment follows the

    # two-stage schedule below. A step that

    # fails is rejected and retried with half the increment; the

    # increment grows back after easy steps.

    # Two-stage schedule: increment_stage1 up to u_switch, then

    # increment_stage2.

    u_switch = steps_stage1 * increment_stage1



    def nominal_increment(u):

        return (

            increment_stage1 if u < u_switch * (1.0 - 1.0e-9)

            else increment_stage2

        )



    du = increment_stage1
    snapshot_targets = sorted(set(float(x) * 1.0e-3 for x in snapshot_targets_mm))
    next_snapshot = 0



    history = {

        "displacement": [],

        "force": [],

        "max_damage": [],

    }

    peak_force = 0.0

    applied = 0.0

    step = 0

    last_written = 0



    xdmf = io.XDMFFile(comm, str(output_path / "fields.xdmf"), "w")

    xdmf.write_mesh(domain)



    def one_pass_solve(target):
        """One mechanical solve followed by one damage solve; not coupled convergence."""
        top_value.value = target
        u_before = u.x.array.copy()
        d_before = d.x.array.copy()
        try:
            mech_problem.solve()
        except PETSc.Error as exc:
            raise StepFailed("SNES raised an error in one_pass.") from exc
        if mech_problem.solver.getConvergedReason() <= 0:
            raise StepFailed(
                f"SNES reason={mech_problem.solver.getConvergedReason()}."
            )
        u.x.scatter_forward()

        psi_cell.interpolate(psi_expression)
        H.x.array[:] = np.maximum(H.x.array, psi_cell.x.array)
        H.x.scatter_forward()
        try:
            damage_problem.solve()
        except PETSc.Error as exc:
            raise StepFailed("Damage linear solve failed in one_pass.") from exc
        if damage_problem.solver.getConvergedReason() <= 0:
            raise StepFailed("Damage KSP did not converge in one_pass.")
        d_trial.x.scatter_forward()
        d.x.array[:] = np.clip(
            np.maximum(d_trial.x.array, d_old.x.array), 0.0, 1.0
        )
        d.x.scatter_forward()
        norm_du = global_l2_norm(u.x.array - u_before, n_u_owned, comm)
        norm_u = global_l2_norm(u.x.array, n_u_owned, comm)
        rel_u = norm_du / max(norm_u, 1.0e-14)
        max_dd = global_max_abs(d.x.array - d_before, n_d_owned, comm)
        # True means both subproblems solved, NOT coupled convergence.
        return True, 1, rel_u, max_dd

    def staggered_solve(target):

        """

        Staggered solve at the displacement `target`.

        Returns (converged, iterations, rel_u, max_dd).

        Raises StepFailed if the mechanical solve fails.

        """

        top_value.value = target

        rel_u = np.inf

        max_dd = np.inf



        relaxation = 1.0
        previous_residual = np.inf
        for it in range(1, max_stagger_iterations + 1):

            u_before = u.x.array.copy()

            d_before = d.x.array.copy()



            # 1) displacement at fixed damage

            try:

                mech_problem.solve()

            except PETSc.Error as exc:

                raise StepFailed("SNES raised an error.") from exc



            reason = mech_problem.solver.getConvergedReason()

            if reason <= 0:

                raise StepFailed(f"SNES reason={reason}.")

            u.x.scatter_forward()



            # 2) history field (irreversible driving force)

            psi_cell.interpolate(psi_expression)

            H.x.array[:] = np.maximum(H.x.array, psi_cell.x.array)

            H.x.scatter_forward()



            # 3) damage at fixed displacement (linear solve)

            try:
                damage_problem.solve()
            except PETSc.Error as exc:
                raise StepFailed("Damage linear solve raised a PETSc error.") from exc

            if damage_problem.solver.getConvergedReason() <= 0:
                raise StepFailed("Damage KSP did not converge.")
            d_trial.x.scatter_forward()



            # Adaptive damping of the damage fixed-point iteration.
            # Use the unrelaxed residual for convergence, so damping
            # cannot falsely declare a step converged.
            candidate = np.clip(
                np.maximum(d_trial.x.array, d_old.x.array), 0.0, 1.0
            )
            damage_residual = global_max_abs(
                candidate - d_before, n_d_owned, comm
            )
            if it >= 4 and damage_residual > 0.95 * previous_residual:
                relaxation = max(0.25, 0.5 * relaxation)
            elif it >= 4 and damage_residual < 0.5 * previous_residual:
                relaxation = min(1.0, 1.25 * relaxation)
            previous_residual = damage_residual

            d.x.array[:] = np.clip(
                d_before + relaxation * (candidate - d_before),
                d_old.x.array, 1.0
            )
            d.x.scatter_forward()

            # 4) staggered convergence check

            norm_du = global_l2_norm(

                u.x.array - u_before, n_u_owned, comm

            )

            norm_u = global_l2_norm(u.x.array, n_u_owned, comm)

            rel_u = norm_du / max(norm_u, 1.0e-14)

            max_dd = global_max_abs(

                d.x.array - d_before, n_d_owned, comm

            )



            # Both the actual iterate change and the unrelaxed
            # damage equation residual must satisfy the tolerance.
            if (rel_u < stagger_tolerance
                    and max_dd < stagger_tolerance
                    and damage_residual < stagger_tolerance):

                return True, it, rel_u, max_dd



        return False, max_stagger_iterations, rel_u, max_dd



    try:

        while applied < max_displacement * (1.0 - 1.0e-12):

            du_nom = nominal_increment(applied)
            if paper_fixed_steps:
                du = du_nom

            du_min = du_nom * min_increment_ratio

            du = min(du, du_nom)



            target = min(applied + du, max_displacement)

            if applied < u_switch * (1.0 - 1.0e-9):
                target = min(target, u_switch)

            # Hit requested crack snapshots exactly where possible.
            while (next_snapshot < len(snapshot_targets)
                   and snapshot_targets[next_snapshot] <= applied + 1e-16):
                next_snapshot += 1
            if (next_snapshot < len(snapshot_targets)
                    and applied < snapshot_targets[next_snapshot] < target):
                target = snapshot_targets[next_snapshot]



            snapshot = (

                u.x.array.copy(),

                d.x.array.copy(),

                H.x.array.copy(),

            )



            failure = None

            converged = False

            stagger_it = 0

            rel_u = np.inf

            max_dd = np.inf



            try:

                converged, stagger_it, rel_u, max_dd = (
                    one_pass_solve(target) if staggered_scheme == "one_pass"
                    else staggered_solve(target)
                )

            except StepFailed as exc:

                failure = str(exc)



            # Strict acceptance: no loosely converged steps.
            if failure is not None or not converged:
                # Restore all evolving fields before retrying or aborting.
                u.x.array[:] = snapshot[0]
                d.x.array[:] = snapshot[1]
                H.x.array[:] = snapshot[2]
                u.x.scatter_forward()
                d.x.scatter_forward()
                H.x.scatter_forward()
                top_value.value = applied

                why = failure or (
                    "stagger did not converge "
                    f"(rel_u={rel_u:.3e}, max_dd={max_dd:.3e})"
                )
                if not paper_fixed_steps and du > du_min * 1.000001:
                    du *= 0.5
                    if rank == 0:
                        print(
                            f"  Rejected step to u={target:.4e} m: {why}. "
                            f"Retry increment={du:.3e} m"
                        )
                    continue
                if rank == 0:
                    print("Stopping safely: all previously accepted CSV/XDMF "
                          "results and crack snapshots remain available.")
                raise RuntimeError(
                    f"Load step to u={target:.6e} m failed at "
                    f"minimum increment {du:.3e} m: {why}. "
                    "No unconverged state was accepted."
                )

            # Accept the step.

            step += 1

            applied = target



            d_old.x.array[:] = d.x.array

            d_old.x.scatter_forward()



            local_reaction = fem.assemble_scalar(reaction_form)

            reaction = comm.allreduce(local_reaction, op=MPI.SUM)



            max_damage = global_max_abs(d.x.array, n_d_owned, comm)



            history["displacement"].append(float(applied))

            history["force"].append(float(reaction))

            history["max_damage"].append(float(max_damage))

            peak_force = max(peak_force, reaction)



            if rank == 0:

                with csv_path.open("a", newline="") as f:

                    csv.writer(f).writerow([

                        step,

                        f"{applied:.9e}",

                        f"{reaction:.9e}",

                        f"{max_damage:.8f}",

                        stagger_it,

                        int(converged),

                        f"{rel_u:.4e}",

                        f"{max_dd:.4e}",

                    ])

                print(

                    f"Step {step:4d} | "

                    f"u={applied:.4e} m "

                    f"({100.0 * applied / max_displacement:5.1f}%) | "

                    f"F={reaction:.4e} N/m | "

                    f"d_max={max_damage:.4f} | "

                    f"stagger={stagger_it}{'' if converged else '*'} | "

                    f"du={du:.2e}"

                )



            is_snapshot = (
                next_snapshot < len(snapshot_targets)
                and abs(applied - snapshot_targets[next_snapshot]) < 1.0e-13
            )
            if step % output_every == 0 or is_snapshot:
                xdmf.write_function(d, float(applied))
                xdmf.write_function(u, float(applied))
                last_written = step

            if is_snapshot:
                if snapshot_callback is not None and comm.size == 1:
                    snapshot_callback(
                        domain, d, applied, reaction, step, output_path
                    )
                next_snapshot += 1



            if (

                peak_force > 0.0

                and reaction < stop_force_fraction * peak_force

            ):

                if rank == 0:

                    print(

                        "Specimen has failed (force dropped below "

                        f"{100.0 * stop_force_fraction:.0f}% of the "

                        "peak). Stopping."

                    )

                break



            # Grow the increment back towards the nominal one: fast

            # after an easy step, slowly otherwise.

            if paper_fixed_steps:
                du = nominal_increment(applied)
            elif converged and stagger_it <= 4:

                du = min(2.0 * du, du_nom)

            else:

                du = min(1.25 * du, du_nom)



        if last_written != step:
            xdmf.write_function(d, float(applied))
            xdmf.write_function(u, float(applied))
            last_written = step

    finally:
        # Preserve the latest ACCEPTED fields even on convergence failure.
        if step > 0 and last_written != step:
            xdmf.write_function(d, float(applied))
            xdmf.write_function(u, float(applied))
        xdmf.close()



    if rank == 0:

        print(f"\nSimulation finished. Results: {csv_path}")



    return {

        "displacement": np.asarray(history["displacement"]),

        "force": np.asarray(history["force"]),

        "max_damage": np.asarray(history["max_damage"]),

        "u": u,

        "d": d,

        "H": H,

    }