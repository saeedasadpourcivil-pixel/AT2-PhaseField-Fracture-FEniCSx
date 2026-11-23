"""
parameters.py
--------------
Parameters for the single-edge-notched tension benchmark of Miehe et al.
(2D AT2 phase-field fracture, plane strain, no plasticity).

Specimen: square plate L x L with a horizontal pre-existing notch that
enters from the left edge at mid-height and reaches the plate centre.
    - bottom edge: fully clamped (ux = uy = 0)
    - top edge   : prescribed vertical displacement (ux free)

All quantities are in SI units. The benchmark data in mm / kN are
converted:
    E  = 210 kN/mm^2  = 210e9 Pa
    Gc = 0.0027 kN/mm = 2.7 N/mm = 2.7e3 J/m^2
    lc = 0.0075 mm    = 7.5e-6 m
    L  = 1 mm         = 1e-3 m
"""

from dataclasses import dataclass
import math


@dataclass
class MaterialParameters:
    """Elastic and fracture properties."""

    # Linear elasticity
    E: float = 210.0e9           # Young's modulus [Pa]
    nu: float = 0.30             # Poisson's ratio [-]

    # Fracture properties
    Gc: float = 2.70e3           # Critical energy release rate [J/m^2]
    ell: float = 7.5e-6          # Phase-field length scale [m]

    # Small residual stiffness to avoid a singular system
    kappa: float = 1.0e-8        # Residual stiffness [-]

    # Tension/compression split of the strain energy:
    #   "spectral"              : Miehe et al. (2010)
    #   "volumetric_deviatoric" : Amor-type split
    energy_split: str = "spectral"


@dataclass
class GeometryParameters:
    """Square plate with a horizontal edge notch."""

    L: float = 1.0e-3            # Width, along x [m]
    H: float = 1.0e-3            # Height, along y (loading direction) [m]
    notch_length: float = 0.5e-3 # Notch length from x = 0 [m]
    notch_width: float = 3.75e-6  # Total width of the notch slit [m]

    # Mesh sizes (paper: hmin = 0.001 mm, hmax = 0.01 mm)
    h_tip: float = 1.0e-6        # Around the notch tip [m]
    h_target: float = 2.0e-6     # Along the crack path [m]
    h_far: float = 1.0e-5        # Far field [m]
    band_half_width: float = 30.0e-6  # Half-width of refined band [m]


@dataclass
class LoadingParameters:
    """Displacement-controlled loading (applied on the top edge, in y)."""

    # Two-stage loading as in the paper:
    #   stage 1: 500 increments of 1e-5 mm (up to 5e-3 mm)
    #   stage 2: increments of 1e-6 mm afterwards
    increment_stage1: float = 1.0e-8  # [m]  (1e-5 mm)
    steps_stage1: int = 500
    increment_stage2: float = 1.0e-9  # [m]  (1e-6 mm)

    # Upper limit of the applied displacement [m] (0.01 mm). The run
    # stops earlier, when the specimen has failed. The solver halves an
    # increment automatically if a step does not converge.
    max_displacement: float = 1.0e-5


@dataclass
class SolverParameters:
    """Numerical settings."""

    petsc_rtol: float = 1.0e-8
    petsc_atol: float = 1.0e-10
    max_iterations: int = 1000


# Create parameter objects
material = MaterialParameters()
geometry = GeometryParameters()
loading = LoadingParameters()
solver = SolverParameters()


def at2_critical_stress():
    """Uniaxial critical stress of the AT2 model (homogeneous state) [Pa]."""
    return math.sqrt(
        27.0 * material.E * material.Gc / (256.0 * material.ell)
    )


def validate_parameters():
    """Check basic physical and numerical parameter constraints."""

    assert material.E > 0.0, "Young's modulus must be positive."
    assert -1.0 < material.nu < 0.5, "Invalid Poisson's ratio."
    assert material.Gc > 0.0, "Fracture energy must be positive."
    assert material.ell > 0.0, "Length scale must be positive."
    assert 0.0 < material.kappa < 1.0, "kappa must be between 0 and 1."
    assert material.energy_split in (
        "spectral", "volumetric_deviatoric"
    ), "energy_split must be 'spectral' or 'volumetric_deviatoric'."

    assert geometry.L > 0.0 and geometry.H > 0.0
    assert 0.0 < geometry.notch_length < geometry.L
    assert 0.0 < geometry.notch_width < geometry.H
    assert geometry.h_tip > 0.0
    assert geometry.h_tip <= geometry.h_target <= geometry.h_far
    assert geometry.band_half_width > 0.0

    assert loading.max_displacement > 0.0
    assert loading.increment_stage1 > 0.0
    assert loading.increment_stage2 > 0.0
    assert loading.steps_stage1 >= 1

    assert solver.petsc_rtol > 0.0
    assert solver.petsc_atol > 0.0
    assert solver.max_iterations >= 1

    # AT2 mesh resolution should resolve the phase-field length scale.
    if geometry.h_target > material.ell / 3.0 * (1.0 + 1.0e-9):
        print(
            "WARNING: h_target is larger than ell/3. "
            "Consider refining the mesh along the crack path."
        )

    if geometry.band_half_width < 3.0 * material.ell:
        print(
            "WARNING: refined band is narrower than 3*ell. "
            "The crack may leave the fine mesh."
        )

    if geometry.notch_width < 2.0 * geometry.h_tip:
        print(
            "WARNING: notch_width is smaller than 2*h_tip; the slit "
            "is poorly resolved."
        )


if __name__ == "__main__":
    validate_parameters()

    mm = 1.0e3
    print("Notched-plate AT2 parameters loaded successfully.")
    print(f"Young's modulus       : {material.E:.3e} Pa")
    print(f"Poisson's ratio       : {material.nu:.3f}")
    print(f"Fracture energy Gc    : {material.Gc:.3e} J/m^2")
    print(f"Phase-field length    : {material.ell:.3e} m "
          f"({material.ell * mm:.4f} mm)")
    print(f"Energy split          : {material.energy_split}")
    print(f"AT2 critical stress   : {at2_critical_stress():.3e} Pa")
    print(f"ell / h_target        : {material.ell / geometry.h_target:.2f}")
    print(f"ell / h_tip           : {material.ell / geometry.h_tip:.2f}")
    print(f"Plate (L x H)         : {geometry.L * mm:.3f} x "
          f"{geometry.H * mm:.3f} mm")
    print(f"Notch length / width  : {geometry.notch_length * mm:.3f} / "
          f"{geometry.notch_width * mm:.4f} mm")
    print(f"Max displacement (y)  : {loading.max_displacement:.3e} m "
          f"({loading.max_displacement * mm:.4f} mm)")
    u_switch = loading.steps_stage1 * loading.increment_stage1
    print(f"Stage 1               : {loading.steps_stage1} steps of "
          f"{loading.increment_stage1:.1e} m (up to "
          f"{u_switch * mm:.4f} mm)")
    print(f"Stage 2               : steps of "
          f"{loading.increment_stage2:.1e} m")
    print("Plasticity            : disabled")