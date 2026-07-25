import autogalaxy as ag
import numpy as np
import pytest

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__gaussian__image_2d_from__matches_lmp():
    lp = ag.lp.Gaussian(ell_comps=(0.1, 0.05), intensity=1.0, sigma=5.0)
    lmp = ag.lmp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__gaussian__convergence_2d_from__matches_lmp():
    mp = ag.mp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__gaussian__deflections_yx_2d_from__matches_lmp():
    mp = ag.mp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__gaussian_gradient__image_2d_from__matches_lmp():
    lp = ag.lp.Gaussian(ell_comps=(0.1, 0.05), intensity=1.0, sigma=5.0)
    lmp = ag.lmp.GaussianGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.5,
        mass_to_light_radius=1.0,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__gaussian_gradient__convergence_2d_from__matches_lmp():
    mp = ag.mp.GaussianGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.5,
        mass_to_light_radius=1.0,
    )
    lmp = ag.lmp.GaussianGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.5,
        mass_to_light_radius=1.0,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__gaussian_gradient__deflections_yx_2d_from__matches_lmp():
    mp = ag.mp.GaussianGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.5,
        mass_to_light_radius=1.0,
    )
    lmp = ag.lmp.GaussianGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=0.5,
        mass_to_light_radius=1.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__sersic__image_2d_from__matches_lmp():
    lp = ag.lp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
    )
    lmp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__sersic__convergence_2d_from__matches_lmp():
    mp = ag.mp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__sersic__deflections_yx_2d_from__matches_lmp():
    mp = ag.mp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__exponential__image_2d_from__matches_lmp():
    lp = ag.lp.Exponential(ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6)
    lmp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__exponential__convergence_2d_from__matches_lmp():
    mp = ag.mp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__exponential__deflections_yx_2d_from__matches_lmp():
    mp = ag.mp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__dev_vaucouleurs__image_2d_from__matches_lmp():
    lp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
    )
    lmp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__dev_vaucouleurs__convergence_2d_from__matches_lmp():
    mp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__dev_vaucouleurs__deflections_yx_2d_from__matches_lmp():
    mp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__sersic_gradient__image_2d_from__matches_lmp():
    lp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
    )
    lmp = ag.lmp.SersicGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid), 1.0e-4
    )


def test__sersic_gradient__convergence_2d_from__matches_lmp():
    mp = ag.lmp.SersicGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    lmp = ag.lmp.SersicGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid), 1.0e-4
    )


def test__sersic_gradient__deflections_yx_2d_from__matches_lmp():
    mp = ag.lmp.SersicGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    lmp = ag.lmp.SersicGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid), 1.0e-4
    )


def test__exponential_gradient__image_2d_from__matches_lmp():
    lp = ag.lmp.Exponential(ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6)
    lmp = ag.lmp.ExponentialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert lp.image_2d_from(grid=grid) == pytest.approx(
        lmp.image_2d_from(grid=grid).array, 1.0e-4
    )


def test__exponential_gradient__convergence_2d_from__matches_lmp():
    mp = ag.lmp.ExponentialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    lmp = ag.lmp.ExponentialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert mp.convergence_2d_from(grid=grid) == pytest.approx(
        lmp.convergence_2d_from(grid=grid).array, 1.0e-4
    )


def test__exponential_gradient__deflections_yx_2d_from__matches_lmp():
    mp = ag.lmp.ExponentialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    lmp = ag.lmp.ExponentialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        lmp.deflections_yx_2d_from(grid=grid).array, 1.0e-4
    )


def test__core_sersic__image_2d_from__matches_lmp():
    lp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05), effective_radius=0.6, sersic_index=2.0
    )
    lmp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert (lp.image_2d_from(grid=grid) == lmp.image_2d_from(grid=grid)).all()


def test__core_sersic__convergence_2d_from__matches_lmp():
    mp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert (
        mp.convergence_2d_from(grid=grid) == lmp.convergence_2d_from(grid=grid)
    ).all()


def test__core_sersic__deflections_yx_2d_from__matches_lmp():
    mp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert (
        mp.deflections_yx_2d_from(grid=grid) == lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__chameleon__image_2d_from__matches_lmp():
    lp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
    )
    lmp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    assert (lp.image_2d_from(grid=grid) == lmp.image_2d_from(grid=grid)).all()


def test__chameleon__convergence_2d_from__matches_lmp():
    mp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    assert (
        mp.convergence_2d_from(grid=grid) == lmp.convergence_2d_from(grid=grid)
    ).all()


def test__chameleon__deflections_yx_2d_from__matches_lmp():
    mp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )
    lmp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    #    assert (mp.potential_2d_from(grid=grid) == lmp.potential_2d_from(grid=grid)).all()
    assert (
        mp.deflections_yx_2d_from(grid=grid) == lmp.deflections_yx_2d_from(grid=grid)
    ).all()
