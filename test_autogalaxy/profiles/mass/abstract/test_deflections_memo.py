import sys
import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.mass.abstract import deflections_memo


@pytest.fixture(autouse=True)
def clear_memo():
    deflections_memo.memo_clear()
    yield
    deflections_memo.memo_clear()


def grid_2d(origin=(0.0, 0.0)):
    return ag.Grid2D.uniform(
        shape_native=(12, 12), pixel_scales=0.2, origin=origin, over_sample_size=1
    )


def gaussian(mass_to_light_ratio=2.0):
    return ag.mp.Gaussian(
        centre=(0.05, -0.05),
        ell_comps=(0.1, 0.15),
        intensity=1.3,
        sigma=0.9,
        mass_to_light_ratio=mass_to_light_ratio,
    )


def values(result):
    return np.asarray(getattr(result, "array", result), dtype=float)


def test__repeated_call_with_same_values_hits__different_values_misses():
    grid = grid_2d()
    profile = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.1), einstein_radius=1.2
    )

    deflections_memo.deflections_yx_2d_from(profile, grid, np)
    assert deflections_memo.memo_stats()["misses"] == 1
    assert deflections_memo.memo_stats()["hits"] == 0

    deflections_memo.deflections_yx_2d_from(profile, grid, np)
    assert deflections_memo.memo_stats()["hits"] == 1

    # An identically parameterised *new* object hits: the key is values, not identity.
    same = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.1), einstein_radius=1.2
    )
    deflections_memo.deflections_yx_2d_from(same, grid, np)
    assert deflections_memo.memo_stats()["hits"] == 2

    moved = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.1), einstein_radius=1.3
    )
    deflections_memo.deflections_yx_2d_from(moved, grid, np)
    assert deflections_memo.memo_stats()["misses"] == 2


def test__l1_hit_returns_the_same_type_and_content_as_the_direct_call():
    grid = grid_2d()
    profile = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.1), einstein_radius=1.2
    )

    direct = profile.deflections_yx_2d_from(grid=grid)

    deflections_memo.deflections_yx_2d_from(profile, grid, np)
    hit = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert type(hit) is type(direct)
    assert values(hit) == pytest.approx(values(direct), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("ratio", [0.5, 3.0])
def test__l2_gaussian_rescale_is_exact_against_the_direct_call(ratio):
    grid = grid_2d()

    # Fill the memo at one ratio, then hit it at another: the stored field is the
    # unit-ratio field, so the second call is a rescale, not a recomputation.
    deflections_memo.deflections_yx_2d_from(gaussian(mass_to_light_ratio=1.7), grid, np)

    profile = gaussian(mass_to_light_ratio=ratio)
    hit = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats()["hits"] == 1
    assert deflections_memo.memo_stats()["misses"] == 1

    direct = profile.deflections_yx_2d_from(grid=grid)

    assert type(hit) is type(direct)
    assert values(hit) == pytest.approx(values(direct), rel=1e-12, abs=0.0)


def test__l2_applies_to_the_light_and_mass_gaussian_too():
    grid = grid_2d()

    def lmp_gaussian(mass_to_light_ratio):
        return ag.lmp.Gaussian(
            centre=(0.05, -0.05),
            ell_comps=(0.1, 0.15),
            intensity=1.3,
            sigma=0.9,
            mass_to_light_ratio=mass_to_light_ratio,
        )

    deflections_memo.deflections_yx_2d_from(lmp_gaussian(1.0), grid, np)

    profile = lmp_gaussian(4.5)
    hit = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats()["hits"] == 1

    direct = profile.deflections_yx_2d_from(grid=grid)

    assert values(hit) == pytest.approx(values(direct), rel=1e-12, abs=0.0)


def test__gaussian_gradient_takes_l1_only():
    grid = grid_2d()

    def gradient(mass_to_light_gradient):
        return ag.mp.GaussianGradient(
            centre=(0.05, -0.05),
            ell_comps=(0.1, 0.15),
            intensity=1.3,
            sigma=0.9,
            mass_to_light_ratio_base=1.5,
            mass_to_light_gradient=mass_to_light_gradient,
            mass_to_light_radius=1.0,
        )

    deflections_memo.deflections_yx_2d_from(gradient(0.5), grid, np)
    deflections_memo.deflections_yx_2d_from(gradient(0.5), grid, np)

    stats = deflections_memo.memo_stats()
    assert (stats["hits"], stats["misses"], stats["entries"]) == (1, 1, 1)

    # An L2 profile would key past the ratio; the gradient must not, so a changed
    # gradient (which changes only the derived mass_to_light_ratio) is a miss.
    profile = gradient(1.5)
    result = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats()["misses"] == 2
    assert values(result) == pytest.approx(
        values(profile.deflections_yx_2d_from(grid=grid)), rel=1e-12, abs=0.0
    )


def test__grid_is_keyed_by_content_not_identity():
    profile = gaussian()

    deflections_memo.deflections_yx_2d_from(profile, grid_2d(), np)

    # A freshly built grid with the same coordinates hits.
    deflections_memo.deflections_yx_2d_from(profile, grid_2d(), np)
    assert deflections_memo.memo_stats()["hits"] == 1

    # A grid built at a different origin misses.
    deflections_memo.deflections_yx_2d_from(profile, grid_2d(origin=(0.5, 0.5)), np)
    assert deflections_memo.memo_stats()["misses"] == 2


def test__irregular_grid_with_the_same_coordinates_does_not_collide_with_the_grid_2d():
    profile = gaussian()
    grid = grid_2d()

    deflections_memo.deflections_yx_2d_from(profile, grid, np)

    irregular = ag.Grid2DIrregular(values=np.array(grid.array))
    result = deflections_memo.deflections_yx_2d_from(profile, irregular, np)

    assert deflections_memo.memo_stats()["hits"] == 0
    assert deflections_memo.memo_stats()["misses"] == 2
    assert type(result) is type(profile.deflections_yx_2d_from(grid=irregular))


def test__kill_switch_disables_the_memo(monkeypatch):
    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO", "0")

    grid = grid_2d()
    profile = gaussian()

    for _ in range(3):
        result = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats() == {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "bytes": 0,
        "entries": 0,
        "jax_folds": 0,
    }
    assert values(result) == pytest.approx(
        values(profile.deflections_yx_2d_from(grid=grid)), rel=1e-12, abs=0.0
    )


def test__memo_disabled_suspends_the_memo_and_restores_it(monkeypatch):
    grid = grid_2d()
    profile = gaussian()

    with deflections_memo.memo_disabled():
        for _ in range(3):
            result = deflections_memo.deflections_yx_2d_from(profile, grid, np)

        assert deflections_memo.memo_stats()["entries"] == 0
        assert values(result) == pytest.approx(
            values(profile.deflections_yx_2d_from(grid=grid)), rel=1e-12, abs=0.0
        )

    deflections_memo.deflections_yx_2d_from(profile, grid, np)
    deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats()["hits"] == 1

    # Nesting is safe, and an exception still restores the previous state.
    with pytest.raises(RuntimeError):
        with deflections_memo.memo_disabled():
            with deflections_memo.memo_disabled():
                assert not deflections_memo.memo_enabled()
            assert not deflections_memo.memo_enabled()
            raise RuntimeError("boom")

    assert deflections_memo.memo_enabled()


def test__byte_cap_evicts_the_oldest_entry(monkeypatch):
    grid = grid_2d()

    one_field_bytes = grid.array.size * 8

    # A cap of one-and-a-bit fields: the second store evicts the first.
    monkeypatch.setenv(
        "AUTOGALAXY_DEFLECTIONS_MEMO_MAX_MB", str(1.5 * one_field_bytes / 1024**2)
    )

    first = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    second = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=2.0)

    deflections_memo.deflections_yx_2d_from(first, grid, np)
    deflections_memo.deflections_yx_2d_from(second, grid, np)

    stats = deflections_memo.memo_stats()
    assert stats["entries"] == 1
    assert stats["evictions"] == 1
    assert stats["bytes"] == one_field_bytes

    # The evicted entry is a miss, never a wrong answer.
    result = deflections_memo.deflections_yx_2d_from(first, grid, np)
    assert deflections_memo.memo_stats()["misses"] == 3
    assert values(result) == pytest.approx(
        values(first.deflections_yx_2d_from(grid=grid)), rel=1e-12, abs=0.0
    )


def test__entry_larger_than_the_whole_cap_is_not_stored(monkeypatch):
    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO_MAX_MB", "0")

    grid = grid_2d()
    profile = gaussian()

    deflections_memo.deflections_yx_2d_from(profile, grid, np)
    deflections_memo.deflections_yx_2d_from(profile, grid, np)

    stats = deflections_memo.memo_stats()
    assert stats["entries"] == 0
    assert stats["bytes"] == 0
    assert stats["hits"] == 0


def test__profile_holding_a_non_scalar_attribute_falls_through():
    grid = grid_2d()

    profile = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    profile.einstein_radius = np.array([1.0, 2.0])

    result = deflections_memo.deflections_yx_2d_from(profile, grid, np)

    assert deflections_memo.memo_stats() == {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "bytes": 0,
        "entries": 0,
        "jax_folds": 0,
    }
    assert result is not None

    # A `Basis`, whose constructor argument is a list of profiles, is unmemoisable in
    # its own right -- its members are memoised individually instead.
    basis = ag.lp_basis.Basis(profile_list=[gaussian()])
    deflections_memo.deflections_yx_2d_from(basis, grid, np)

    assert deflections_memo.memo_stats()["entries"] == 1


def test__galaxy_sum_matches_the_sum_with_the_memo_off(monkeypatch):
    grid = grid_2d()

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.1), einstein_radius=1.2
        ),
        stellar=gaussian(mass_to_light_ratio=2.5),
    )

    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO", "0")
    off = galaxy.deflections_yx_2d_from(grid=grid)

    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO", "1")
    galaxy.deflections_yx_2d_from(grid=grid)
    on = galaxy.deflections_yx_2d_from(grid=grid)

    assert deflections_memo.memo_stats()["hits"] == 2
    assert values(on) == pytest.approx(values(off), rel=1e-12, abs=0.0)


def test__basis_sum_matches_the_sum_with_the_memo_off(monkeypatch):
    grid = grid_2d()

    def basis(mass_to_light_ratio):
        return ag.lp_basis.Basis(
            profile_list=[
                ag.lmp.Gaussian(
                    centre=(0.0, 0.0),
                    ell_comps=(0.05, 0.1),
                    intensity=1.0,
                    sigma=sigma,
                    mass_to_light_ratio=mass_to_light_ratio,
                )
                for sigma in (0.3, 0.9, 2.1)
            ]
        )

    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO", "0")
    off = basis(2.5).deflections_yx_2d_from(grid=grid)

    monkeypatch.setenv("AUTOGALAXY_DEFLECTIONS_MEMO", "1")
    basis(1.0).deflections_yx_2d_from(grid=grid)
    on = basis(2.5).deflections_yx_2d_from(grid=grid)

    assert deflections_memo.memo_stats()["hits"] == 3
    assert deflections_memo.memo_stats()["misses"] == 3
    assert values(on) == pytest.approx(values(off), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Phase 2 -- the JAX trace-time fold, tested numpy-side only.
#
# `test_autogalaxy` never imports jax, so these exercise the parts of the fold that
# are backend-agnostic: the concreteness test, the numpy twin, and the guarantee that
# a numpy caller never enters the fold at all.
# ---------------------------------------------------------------------------


class _FakeTracer:
    """
    A stand-in for a JAX tracer: an object whose *type* is defined in `jax._src.core`,
    which is how the memo tells a traced value apart without importing jax.
    """

    ndim = 2


_FakeTracer.__module__ = "jax._src.core"


def test__is_concrete_array__ndarray_yes__list_and_tracer_no():
    assert deflections_memo._is_concrete_array(np.zeros((3, 2)), xp=np) is True

    # Not an array at all.
    assert deflections_memo._is_concrete_array([[0.0, 0.0]], xp=np) is False
    assert deflections_memo._is_concrete_array(3.0, xp=np) is False

    tracer = _FakeTracer()

    # A jax-typed object is rejected on the numpy backend by the `xp is np` gate, and on
    # a non-numpy backend by the `isinstance(..., jax.Array)` test (or, in a process
    # that never imported jax, by the `sys.modules` lookup failing outright). Neither
    # route imports jax itself -- other tests in this suite build `use_jax` analyses, so
    # whether jax is already loaded is not this test's to assert, only that this call
    # does not change it.
    jax_was_imported = "jax" in sys.modules

    assert deflections_memo._is_concrete_array(tracer, xp=np) is False
    assert deflections_memo._is_concrete_array(tracer, xp=object()) is False
    assert ("jax" in sys.modules) is jax_was_imported


def test__numpy_twin_reproduces_the_coordinates_and_the_mask():
    grid = grid_2d()

    twin = deflections_memo._numpy_twin_from(grid, np.array(grid.array))

    assert type(twin) is type(grid)
    assert twin.array == pytest.approx(np.array(grid.array), rel=0.0, abs=0.0)
    assert (np.asarray(twin.mask) == np.asarray(grid.mask)).all()
    assert twin.pixel_scales == grid.pixel_scales

    irregular = ag.Grid2DIrregular(values=np.array(grid.array))

    twin = deflections_memo._numpy_twin_from(irregular, np.array(irregular.array))

    assert type(twin) is type(irregular)
    assert twin.array == pytest.approx(np.array(irregular.array), rel=0.0, abs=0.0)

    # A grid class the memo cannot rebuild is not guessed at.
    assert deflections_memo._numpy_twin_from(object(), np.zeros((2, 2))) is None


def test__numpy_backend_never_folds__jax_folds_counter_stays_zero():
    grid = grid_2d()
    profile = gaussian()

    for _ in range(3):
        deflections_memo.deflections_yx_2d_from(profile, grid, np)

    stats = deflections_memo.memo_stats()

    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["jax_folds"] == 0

    # An L1 profile too, and the fingerprint helper's twin is None for numpy -- a numpy
    # grid is its own twin, and holding it here would defeat the weakref cache.
    isothermal = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_memo.deflections_yx_2d_from(isothermal, grid, np)

    assert deflections_memo.memo_stats()["jax_folds"] == 0

    fingerprint, twin = deflections_memo._grid_fingerprint_and_twin(grid, xp=np)

    assert fingerprint is not None
    assert twin is None


def test__concrete_scalar_value__reads_back_nothing_on_the_numpy_backend():
    assert deflections_memo._concrete_scalar_value(1.5, xp=np) is None
    assert deflections_memo._concrete_scalar_value(np.array(1.5), xp=np) is None
    assert deflections_memo._concrete_scalar_value(_FakeTracer(), xp=np) is None

    # A float and a 0-d value of the same number share one token, so a numpy and a JAX
    # caller of the same fixed geometry land on the same memo entry. `np.float64`
    # subclasses `float`, so it shares that token too; a `np.float32` does not and keeps
    # its own dtype-tagged token, because it is a different number.
    assert deflections_memo._scalar_token(1.5) == "float:1.5"
    assert deflections_memo._scalar_token(np.float64(1.5)) == "float:1.5"
    assert deflections_memo._scalar_token(np.float32(1.5)) == "np:<f4:1.5"
    assert deflections_memo._scalar_token(3) == "int:3"
