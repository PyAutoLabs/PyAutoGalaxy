"""
The domain semantics a model figure draws, asserted over the **real** classes.

PyAutoFit cannot know which parameters a fit solves rather than samples; phase 3
of the ``model-figures`` epic supplies that from here, as a class attribute
(``__solved_parameters__``) that PyAutoFit's ``graph_spec`` reads.  These tests
pin the semantics end to end: the class declares, ``GraphSpec`` extracts, and
``ModelPlotter(...).presentation()`` says the right thing on the pill.

Every assertion is over a real PyAutoGalaxy class -- the structural doubles in
``test_autofit`` cannot catch a declaration that was never made.
"""

import matplotlib

matplotlib.use("Agg")

import autofit as af
import autogalaxy as ag


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def rows_of(spec):
    """Every ``ParamRow`` in a spec, keyed by its dotted path."""
    rows = {}

    def walk(node):
        for row in node.rows:
            rows[".".join(row.path)] = row
        for child in node.children:
            walk(child)

    walk(spec.root)
    return rows


def named(spec, name):
    """Every row with this exact attribute name, in walk order."""
    return [row for path, row in rows_of(spec).items() if row.name == name]


def pill_texts(presentation):
    return [pill.text for pill in presentation.pills()]


def card_of(presentation, title_fragment):
    for card in presentation.walk():
        if title_fragment in card.title:
            return card
    raise AssertionError(
        f"no card whose title contains {title_fragment!r}: "
        f"{[card.title for card in presentation.walk()]}"
    )


# ----------------------------------------------------------------------------
# linear light profiles -- `intensity` is solved by the inversion
# ----------------------------------------------------------------------------


def test__linear_light_profile_intensity_is_solved():
    spec = af.GraphSpec.from_model(af.Model(ag.lp_linear.Sersic))

    row = rows_of(spec)["intensity"]

    assert row.sampling == "solved"
    assert row.in_model_info is False
    assert row.provenance.kind == "solved-by-fit"

    presentation = af.ModelPlotter(af.Model(ag.lp_linear.Sersic)).presentation()

    assert "intensity · solved" in pill_texts(presentation)


def test__standard_light_profile_intensity_is_free():
    spec = af.GraphSpec.from_model(af.Model(ag.lp.Sersic))

    row = rows_of(spec)["intensity"]

    assert row.sampling == "free"
    assert row.in_model_info is True

    assert "intensity" in pill_texts(
        af.ModelPlotter(af.Model(ag.lp.Sersic)).presentation()
    )


def test__solved_declaration_is_inherited_by_every_linear_subclass():
    for cls in (
        ag.lp_linear.Gaussian,
        ag.lp_linear.Exponential,
        ag.lp_linear.SersicCore,
    ):
        spec = af.GraphSpec.from_model(af.Model(cls))

        assert rows_of(spec)["intensity"].sampling == "solved", cls.__name__


# ----------------------------------------------------------------------------
# `Basis` -- the solved intensities belong to the members, not the basis
# ----------------------------------------------------------------------------


def test__basis_has_no_solved_intensity_of_its_own__members_do():
    model = af.Model(
        ag.lp_basis.Basis,
        profile_list=[af.Model(ag.lp_linear.Gaussian) for _ in range(3)],
    )

    spec = af.GraphSpec.from_model(model, collapse=False)

    assert [row.name for row in spec.root.rows if row.name == "intensity"] == []

    intensities = named(spec, "intensity")

    assert len(intensities) == 3
    assert {row.sampling for row in intensities} == {"solved"}


def test__basis_members_collapse_into_one_plate_whose_repeats_name_the_solved_intensity():
    model = af.Model(
        ag.lp_basis.Basis,
        profile_list=[af.Model(ag.lp_linear.Gaussian) for _ in range(3)],
    )

    spec = af.GraphSpec.from_model(model)

    plates = [
        node
        for node in _walk_nodes(spec.root)
        if node.plate is not None and node.plate.count == 3
    ]

    assert len(plates) == 1
    assert any("intensity solved" in line for line in plates[0].plate.repeats)

    presentation = af.ModelPlotter(model).presentation()

    assert "intensity · solved" in pill_texts(presentation)


def _walk_nodes(node):
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


# ----------------------------------------------------------------------------
# point sources -- `PointSolved` solves its centre, `Point` samples it
# ----------------------------------------------------------------------------


def test__point_solved_has_exactly_one_solved_centre_row():
    spec = af.GraphSpec.from_model(af.Model(ag.ps.PointSolved))

    assert [row.name for row in spec.root.rows] == ["centre"]

    row = spec.root.rows[0]

    assert row.sampling == "solved"
    assert row.in_model_info is False

    presentation = af.ModelPlotter(af.Model(ag.ps.PointSolved)).presentation()

    assert "centre · solved" in pill_texts(presentation)


def test__point_centre_is_a_free_tuple():
    spec = af.GraphSpec.from_model(af.Model(ag.ps.Point))

    row = rows_of(spec)["centre"]

    assert row.sampling == "free"
    assert row.dimensionality == "tuple"
    assert row.in_model_info is True


# ----------------------------------------------------------------------------
# the `missing` state -- unset configuration on a `Delaunay` mesh
# ----------------------------------------------------------------------------


def test__delaunay_areas_factor_has_no_prior_configured_and_is_missing():
    spec = af.GraphSpec.from_model(af.Model(ag.mesh.Delaunay))

    row = rows_of(spec)["areas_factor"]

    assert row.sampling == "missing"
    assert row.prior_cls_name == "ConfigException"
    assert spec.counts["missing"] == 1

    presentation = af.ModelPlotter(af.Model(ag.mesh.Delaunay)).presentation()

    assert "areas_factor · missing" in pill_texts(presentation)


def test__delaunay_pixels_and_zeroed_pixels_are_fixed_not_missing():
    """
    Measured, not assumed.

    The epic's brief expected ``pixels``, ``zeroed_pixels`` and ``areas_factor``
    to be three ``missing`` rows.  They are not: ``pixels: int`` has no default,
    so ``af.Model`` fills the slot with ``af.Model(int)`` -- a *fixed* row by
    rule R7, never a ``ConfigException`` -- and ``zeroed_pixels: Optional[int] =
    0`` simply takes its default.  Only ``areas_factor`` is genuinely unset
    configuration.  This test pins that distinction so the figure cannot start
    calling an unset int "missing" without someone noticing.
    """
    spec = af.GraphSpec.from_model(af.Model(ag.mesh.Delaunay))

    rows = rows_of(spec)

    assert rows["pixels"].sampling == "fixed"
    assert [row.name for row in spec.root.rows] == ["pixels", "areas_factor"]

    set_spec = af.GraphSpec.from_model(
        af.Model(ag.mesh.Delaunay, pixels=500, zeroed_pixels=0)
    )
    set_rows = set_spec.root.rows

    assert [row.name for row in set_rows] == ["pixels", "zeroed_pixels", "areas_factor"]
    assert [row.sampling for row in set_rows] == ["fixed", "fixed", "missing"]
    assert set_spec.counts["missing"] == 1


# ----------------------------------------------------------------------------
# pixelization -- the source reconstruction is solved by the inversion
# ----------------------------------------------------------------------------


def pixelization_model():
    return af.Model(
        ag.Pixelization,
        mesh=af.Model(ag.mesh.Delaunay, pixels=500, zeroed_pixels=0),
        regularization=af.Model(ag.reg.ConstantSplit),
    )


def test__pixelization_reconstruction_is_solved_and_regularization_stays_free():
    model = pixelization_model()

    spec = af.GraphSpec.from_model(model)
    rows = rows_of(spec)

    reconstruction = rows["reconstruction"]

    assert reconstruction.sampling == "solved"
    assert reconstruction.in_model_info is False
    assert reconstruction.provenance.kind == "solved-by-fit"

    assert rows["regularization.coefficient"].sampling == "free"
    assert spec.counts["unique_sampled_scalars"] == model.prior_count

    presentation = af.ModelPlotter(model).presentation()

    assert "reconstruction · solved" in [
        pill.text for pill in card_of(presentation, "Pixelization").pills
    ]


# ----------------------------------------------------------------------------
# redshift -- free is an ordinary pill, fixed is a subtitle
# ----------------------------------------------------------------------------


def free_redshift_galaxy():
    redshift = af.Model(ag.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    return af.Model(ag.Galaxy, redshift=redshift, bulge=af.Model(ag.lp.Sersic))


def test__free_redshift_is_a_pill_on_the_galaxy_card_not_a_child_card():
    model = free_redshift_galaxy()

    presentation = af.ModelPlotter(model).presentation()
    galaxy = card_of(presentation, "Galaxy")

    assert "redshift" in [pill.text for pill in galaxy.pills]
    assert "Redshift" not in [child.title for child in galaxy.children]

    spec = af.GraphSpec.from_model(model)

    assert rows_of(spec)["redshift"].sampling == "free"


def test__fixed_redshift_is_the_card_subtitle():
    model = af.Model(ag.Galaxy, redshift=0.5, bulge=af.Model(ag.lp.Sersic))

    presentation = af.ModelPlotter(model).presentation()
    galaxy = card_of(presentation, "Galaxy")

    assert galaxy.subtitle == "redshift = 0.5"
    assert "redshift" not in [pill.text for pill in galaxy.pills]
