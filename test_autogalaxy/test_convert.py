import numpy as np
import pytest
import autofit as af

import autogalaxy as ag


def test__ell_comps_from():
    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.00050025012, angle=0.0)

    assert ell_comps == pytest.approx((0.0, 0.999), abs=1.0e-4)


def test__axis_ratio_and_angle_from():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, 1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(-45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(112.5, abs=1.0e-4)


def test__shear_gamma_1_2_from():
    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=0.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=90.0)

    assert gamma_1 == pytest.approx(-0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=135.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=180.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=225.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=-45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)


def test__shear_magnitude_and_angle_from():
    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=-0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(135.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.07071067811865, abs=1.0e-4)
    assert angle == pytest.approx(-22.5, abs=1.0e-4)


def test__multipole_k_m_and_phi_m_from():
    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(90.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.0, 0.1), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(0.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=2)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(45.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(
        multipole_comps=(-0.1, -0.1), m=2
    )

    assert k_m == pytest.approx(0.14142135, abs=1e-3)
    assert phi == pytest.approx(112.5, abs=1e-3)


def test__multipole_comps_from():
    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=90.0, m=1)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=0.0, m=1)

    assert multipole_comps == pytest.approx((0.0, 0.1), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=45.0, m=2)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.14142135, phi_m=112.5, m=2)

    assert multipole_comps == pytest.approx((-0.1, -0.1), abs=1e-3)


def test__polar_conversions__numpy_origin_values_unchanged():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, 0.0))

    assert axis_ratio == 1.0
    assert angle == 0.0

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.0
    )

    assert magnitude == 0.0
    assert angle == 0.0

    k_m, phi_m = ag.convert.multipole_k_m_and_phi_m_from(
        multipole_comps=(0.0, 0.0), m=4
    )

    assert k_m == 0.0
    assert phi_m == 0.0


def _deflection_objectives():
    """
    Scalar objectives (weighted sums of deflections on a tiny grid) of the profiles whose polar conversions take a
    square root of their components, as functions of those two components.
    """
    import jax.numpy as jnp
    import numpy as np

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.3)
    weights = jnp.asarray(np.random.default_rng(1).normal(size=(9, 2)))

    def shear(c):
        mass = ag.mp.ExternalShear(gamma_1=c[0], gamma_2=c[1])
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    def multipole(c):
        mass = ag.mp.PowerLawMultipole(
            centre=(0.01, 0.02),
            einstein_radius=1.0,
            slope=2.0,
            m=4,
            multipole_comps=(c[0], c[1]),
        )
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    def isothermal(c):
        mass = ag.mp.Isothermal(
            centre=(0.01, 0.02), ell_comps=(c[0], c[1]), einstein_radius=1.0
        )
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    return {"shear": shear, "multipole": multipole, "isothermal": isothermal}


@pytest.mark.parametrize("name", ["shear", "multipole"])
def test__polar_conversions__jax_grad_finite_at_origin_fp64(name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(True):
        f = _deflection_objectives()[name]
        grad = jax.grad(f)(jnp.zeros(2))

        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("name", ["shear", "multipole", "isothermal"])
def test__polar_conversions__jax_grad_finite_at_origin_fp32(name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(False):
        f = _deflection_objectives()[name]
        grad = jax.grad(f)(jnp.zeros(2, dtype=jnp.float32))

        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("name", ["shear", "multipole"])
def test__polar_conversions__jax_grad_at_origin_matches_finite_difference(name):
    """
    Shear and multipole deflections are linear in their components, so the gradient at the origin is well defined
    and equal to the central finite difference there.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(True):
        f = _deflection_objectives()[name]
        grad = np.asarray(jax.grad(f)(jnp.zeros(2)))

        h = 1.0e-5
        fd = np.array(
            [
                (f(jnp.array([h, 0.0])) - f(jnp.array([-h, 0.0]))) / (2 * h),
                (f(jnp.array([0.0, h])) - f(jnp.array([0.0, -h]))) / (2 * h),
            ]
        )

        assert grad == pytest.approx(fd, abs=1.0e-6)


def test__latent_parameter_conversions__invert_the_sampled_components():
    # The functions each profile class declares to PyAutoFit's latent-parameter
    # protocol, vectorised over samples.
    gamma_1 = np.array([0.1, 0.0])
    gamma_2 = np.array([0.0, 0.1])
    latent = ag.convert.latent_shear_from(gamma_1=gamma_1, gamma_2=gamma_2)
    assert latent["magnitude"] == pytest.approx([0.1, 0.1])
    assert latent["angle"].values == pytest.approx([0.0, 45.0])
    assert latent["angle"].period == 180.0

    ell = ag.convert.ell_comps_from(axis_ratio=0.5, angle=30.0)
    latent = ag.convert.latent_ellipse_from(
        ell_comps_0=np.array([ell[0]]), ell_comps_1=np.array([ell[1]])
    )
    assert latent["axis_ratio"] == pytest.approx([0.5])
    assert latent["angle"].values == pytest.approx([30.0])

    comps = ag.convert.multipole_comps_from(k_m=0.05, phi_m=10.0, m=4)
    latent = ag.convert.latent_multipole_from(
        multipole_comps_0=np.array([comps[0]]),
        multipole_comps_1=np.array([comps[1]]),
        m=4.0,
    )
    assert latent["k_m"] == pytest.approx([0.05])
    assert latent["phi_m"].values == pytest.approx([10.0])
    assert latent["phi_m"].period == pytest.approx(90.0)


def test__latent_parameters__declared_on_the_profile_classes():
    from autofit.mapper.prior_model.latent_parameters import latent_parameters_for_class

    assert ("ell_comps_0", "ell_comps_1") in latent_parameters_for_class(
        ag.mp.Isothermal
    )
    assert ("gamma_1", "gamma_2") in latent_parameters_for_class(ag.mp.ExternalShear)
    multipole = latent_parameters_for_class(ag.mp.PowerLawMultipole)
    assert ("ell_comps_0", "ell_comps_1") in multipole
    assert ("multipole_comps_0", "multipole_comps_1") in multipole
    assert (
        "scaled_multipole_comps_0",
        "scaled_multipole_comps_1",
    ) in latent_parameters_for_class(ag.EllipseMultipoleScaled)

    model = af.Model(ag.mp.ExternalShear)
    latent = model.latent_parameters_from(np.array([[0.1, 0.0]]))
    assert latent.names == ["magnitude", "angle"]
    assert latent.parameter_array[0] == pytest.approx([0.1, 0.0])


def test__latent_parameters__angle_posterior_at_the_branch_cut_is_not_split():
    # A shear whose angle posterior peaks at the edge of the conversion's range
    # (near 135 / -45): without recentring the corner plot shows two peaks. The
    # recentring is for plots only (`recentre=True`); the latent output keeps the
    # conversion's range.
    model = af.Model(ag.mp.ExternalShear)
    angles = np.array([130.0, 134.0, -44.0, -40.0])
    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=angles)
    parameter_array = np.column_stack([gamma_1, gamma_2])

    latent = model.latent_parameters_from(parameter_array, recentre=True)

    assert np.ptp(latent.parameter_array[:, 1]) == pytest.approx(10.0)

    latent = model.latent_parameters_from(parameter_array)

    assert latent.parameter_array[:, 1] == pytest.approx(angles)


def test__latent_parameters__shared_mass_ellipticity_retains_owner_latents():
    mass = af.Model(ag.mp.PowerLaw)
    multipole = af.Model(ag.mp.PowerLawMultipole)
    multipole.ell_comps = mass.ell_comps
    multipole.einstein_radius = mass.einstein_radius
    multipole.slope = mass.slope
    multipole.m = 4
    model = af.Collection(mass=mass, multipole=multipole)
    ell = ag.convert.ell_comps_from(axis_ratio=0.5, angle=30.0)
    comps = ag.convert.multipole_comps_from(k_m=0.05, phi_m=10.0, m=4)
    values = {
        "ell_comps_0": ell[0],
        "ell_comps_1": ell[1],
        "multipole_comps_0": comps[0],
        "multipole_comps_1": comps[1],
    }
    samples = np.array(
        [[values.get(path[-1], 1.0) for path in model.unique_prior_paths]]
    )
    latent = model.latent_parameters_from(samples)
    columns = dict(zip(latent.paths, latent.parameter_array[0]))
    # PyAutoFit assigns shared priors to their canonical owner, which can be
    # the multipole; the mass ellipticity conversion must survive either way.
    owner = next(
        path[:-2] for path in model.unique_prior_paths if path[-1] == "ell_comps_0"
    )
    assert columns[owner + ("axis_ratio",)] == pytest.approx(0.5)
    assert columns[owner + ("angle",)] == pytest.approx(30.0)
    assert columns[("multipole", "k_m")] == pytest.approx(0.05)
    assert columns[("multipole", "phi_m")] == pytest.approx(10.0)
    assert latent.names.count("axis_ratio") == 1
    assert not any("ell_comps" in name for name in latent.names)


def test__latent_parameters__scaled_and_light_multipole_values_and_periods():
    c3 = ag.convert.multipole_comps_from(
        k_m=np.array([0.03, 0.06]), phi_m=np.array([10.0, -20.0]), m=3
    )
    c4 = ag.convert.multipole_comps_from(
        k_m=np.array([0.04, 0.08]), phi_m=np.array([15.0, -25.0]), m=4
    )
    scaled = ag.convert.latent_scaled_multipole_from(*c3, m=3)
    assert scaled["k_m"] == pytest.approx([0.03, 0.06])
    assert scaled["phi_m"].values == pytest.approx([10.0, -20.0])
    assert scaled["phi_m"].period == 120.0
    # The light profile evaluates c*cos(m*theta) + s*sin(m*theta).
    light3 = (
        np.array([0.03, 0.06]) * np.cos(np.deg2rad(3 * np.array([10.0, -20.0]))),
        np.array([0.03, 0.06]) * np.sin(np.deg2rad(3 * np.array([10.0, -20.0]))),
    )
    light4 = (
        np.array([0.04, 0.08]) * np.cos(np.deg2rad(4 * np.array([15.0, -25.0]))),
        np.array([0.04, 0.08]) * np.sin(np.deg2rad(4 * np.array([15.0, -25.0]))),
    )
    light = ag.convert.latent_light_multipoles_from(*light3, *light4)
    assert light["k_3"] == pytest.approx([0.03, 0.06])
    assert light["k_4"] == pytest.approx([0.04, 0.08])
    assert light["phi_3"].values == pytest.approx([10.0, -20.0])
    assert light["phi_4"].values == pytest.approx([15.0, 65.0])
    assert light["phi_3"].period == 120.0
    assert light["phi_4"].period == 90.0


def test__latent_declarations__preserve_profile_class_documentation():
    assert "Constant external shear" in ag.mp.ExternalShear.__doc__
    assert "Angular multipole perturbation" in ag.mp.PowerLawMultipole.__doc__


def test__light_multipole_latents__reconstruct_the_radius_perturbation():
    c3, s3, c4, s4 = 0.03, -0.02, 0.04, 0.01
    latent = ag.convert.latent_light_multipoles_from(c3, s3, c4, s4)
    theta = np.linspace(-np.pi, np.pi, 101)
    original = (
        c3 * np.cos(3 * theta)
        + s3 * np.sin(3 * theta)
        + c4 * np.cos(4 * theta)
        + s4 * np.sin(4 * theta)
    )
    reconstructed = sum(
        latent[f"k_{m}"] * np.cos(m * (theta - np.deg2rad(latent[f"phi_{m}"].values)))
        for m in (3, 4)
    )
    assert reconstructed == pytest.approx(original)
    cosine = ag.convert.latent_light_multipoles_from(0.03, 0.0, 0.04, 0.0)
    assert cosine["phi_3"].values == pytest.approx(0.0)
    assert cosine["phi_4"].values == pytest.approx(0.0)


@pytest.mark.parametrize("free_mode", [3, 4])
def test__light_multipole_latents__convert_independent_free_modes(free_mode):
    model = af.Model(ag.lp.SersicMultipole)
    setattr(model, f"multipole_{7 - free_mode}_comps", (0.0, 0.0))
    pair = getattr(model, f"multipole_{free_mode}_comps")
    setattr(pair, f"multipole_{free_mode}_comps_0", af.UniformPrior(-0.1, 0.1))
    setattr(pair, f"multipole_{free_mode}_comps_1", af.UniformPrior(-0.1, 0.1))
    values = {
        f"multipole_{free_mode}_comps_0": 0.03,
        f"multipole_{free_mode}_comps_1": 0.0,
    }
    samples = np.array(
        [[values.get(path[-1], 0.1) for path in model.unique_prior_paths]]
    )
    latent = model.latent_parameters_from(samples)
    columns = dict(zip(latent.names, latent.parameter_array[0]))
    assert columns[f"k_{free_mode}"] == pytest.approx(0.03)
    assert columns[f"phi_{free_mode}"] == pytest.approx(0.0)
    assert not any(f"multipole_{free_mode}_comps" in name for name in latent.names)
