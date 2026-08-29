"""
Model-composition gate for the ``*Power`` regularization siblings added in PyAutoArray.

These schemes are re-exported as ``ag.reg.*`` and carry their own prior configs
(``autogalaxy/config/priors/regularization/*_power.yaml``). Composing them proves both the re-export
chain and that the prior config resolves — including ``power``, which is fixed as a ``Constant``
prior so a non-linear search never samples it.
"""

import autofit as af
import autogalaxy as ag


def test__power_classes_are_re_exported():
    assert ag.reg.AdaptPower is not ag.reg.Adapt
    assert ag.reg.AdaptSplitPower is not ag.reg.AdaptSplit
    assert ag.reg.AdaptSplitZerothPower is not ag.reg.AdaptSplitZeroth
    assert ag.reg.MaternAdaptPowerKernel is not ag.reg.MaternAdaptKernel


def test__model_composition__power_is_a_constant_and_is_not_sampled():
    model = af.Model(ag.reg.AdaptSplitPower)

    assert "power" not in [prior_tuple[0] for prior_tuple in model.prior_tuples]
    assert model.instance_from_prior_medians().power == 1.0

    assert set(prior_tuple[0] for prior_tuple in model.prior_tuples) == {
        "inner_coefficient",
        "outer_coefficient",
        "signal_scale",
    }


def test__model_identifier__differs_from_the_legacy_class():
    identifier_legacy = af.Model(ag.reg.AdaptSplit).identifier
    identifier_power = af.Model(ag.reg.AdaptSplitPower).identifier

    assert identifier_legacy != identifier_power
