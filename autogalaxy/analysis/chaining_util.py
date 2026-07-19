from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from autonerves import conf
from autonerves.dictable import from_json, output_to_json

import autoarray as aa
import autofit as af

if TYPE_CHECKING:
    from autogalaxy.analysis.result import Result

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.basis import Basis

from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles.light import linear as lp_linear
from autogalaxy.profiles import light_and_mass_profiles as lmp
from autogalaxy.profiles import light_linear_and_mass_profiles as lmp_linear


def mass_from(mass, mass_result, unfix_mass_centre: bool = False) -> af.Model:
    """
    Returns an updated mass `Model` whose priors are initialized from a previous results in a pipeline.

    It includes an option to unfix the input `mass_centre` used previously (e.g. in the SLaM SOURCE PIPELINE), such
    that if the `mass_centre` were fixed (e.g. to (0.0", 0.0")) it becomes a free parameter.

    This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    mass
        The mass profile about to be fitted, whose priors are updated based on the previous results.
    mass_result
        The mass profile inferred as a result of the previous pipeline, whose priors are used to update the
        input mass.
    unfix_mass_centre
        If the `mass_centre` was fixed to an input value in a previous pipeline, then `True` will unfix it and make it
        free parameters that are fitted for.

    Returns
    -------
    af.Model(mp.MassProfile)
        The total mass profile whose priors are initialized from a previous result.
    """

    mass.take_attributes(source=mass_result)

    if unfix_mass_centre and isinstance(mass.centre, tuple):
        centre_tuple = mass.centre

        mass.centre = af.Model(mass.cls).centre

        mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

    return mass


def source_custom_model_from(result: Result, source_is_model: bool = False) -> af.Model:
    """
    Setup the source model using the previous pipeline's source result.

    The source light model is not specified by the MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default behaviour is
    to return parametric sources as a model (give they must be updated to properly compute a new mass model) and
    return inversions as an instance (as they have sufficient flexibility to typically not required updating).

    Parameters
    ----------
    result
        The result of the previous source pipeline.
    source_is_model
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        search result it is loaded from. If `False`, it is an instance of that search's result.

    Returns
    -------
    af.Model(Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `source_is_model`.
    """

    redshift = result.instance.galaxies.source.redshift

    if not hasattr(result.instance.galaxies.source, "pixelization"):
        if source_is_model:
            return af.Model(
                Galaxy,
                redshift=redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
            )

        return af.Model(
            Galaxy,
            redshift=redshift,
            bulge=result.instance.galaxies.source.bulge,
            disk=result.instance.galaxies.source.disk,
        )

    if source_is_model:
        pixelization = af.Model(
            aa.Pixelization,
            mesh=result.instance.galaxies.source.pixelization.mesh,
            regularization=result.model.galaxies.source.pixelization.regularization,
        )

        return af.Model(
            Galaxy,
            redshift=redshift,
            pixelization=pixelization,
        )

    pixelization = af.Model(
        aa.Pixelization,
        mesh=result.instance.galaxies.source.pixelization.mesh,
        regularization=result.instance.galaxies.source.pixelization.regularization,
    )

    return af.Model(
        Galaxy,
        redshift=redshift,
        pixelization=pixelization,
    )


def source_from(
    result: Result,
) -> af.Model:
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source_from` method.

    Parameters
    ----------
    result
        The result of the previous source pipeline.

    Returns
    -------
    af.Model(Galaxy)
        The source galaxy with its light profile(s) and pixelization, set up as a model or instance according to
        the input `result`.
    """

    if hasattr(result.instance.galaxies.source, "pixelization"):
        if result.instance.galaxies.source.pixelization is not None:
            return source_custom_model_from(result=result, source_is_model=False)
    return source_custom_model_from(result=result, source_is_model=True)


def extra_galaxies_from(
    result: Result,
    light_as_model: bool = False,
    mass_as_model: bool = False,
    free_centre: bool = False,
) -> af.Collection:
    """
    The extra galaxies API models the light and / or mass of additional galaxies surrouding the main galaxy or
    strong lens system.

    This function performs model composition of extra galaxies for fits using the search chaining API. It makes it
    possible to pass the extra galaxy parameters from a previous search to a new search, such that the new extra
    galaxies are either treated as an instance or model component.

    This function currently requires that mass profiles are `IsothermalSph` objects and that light profiles are
    `Sersic` objects. This will be generalised in the future.

    Parameters
    ----------
    result
        The result, which includes extra galaxies, of the previous search, which via prior passing are used to create
        the new extra galaxy model.
    light_as_model
        If `True`, the extra_galaxy light is passed as a model component, else it is a fixed instance.
    mass_as_model
        If `True`, the extra_galaxy mass is passed as a model component, else it is a fixed instance.
    free_centre
        If `True`, the requested mass and/or light model centres are passed as a model, else they are fixed.

    Returns
    -------
    af.Collection
        A collection of extra galaxy `Galaxy` objects, where each is either an instance or model component.
    """
    # ideal API:

    # extra_galaxies = result.instance.extra_galaxies.as_model((LightProfile, mp.MassProfile,), fixed="centre", prior_pass=True)

    if not hasattr(result.instance, "extra_galaxies"):
        return None

    if result.instance.extra_galaxies is None:
        return None

    if mass_as_model:
        extra_galaxies = result.instance.extra_galaxies.as_model((MassProfile,))

        for extra_galaxy_index in range(len(result.instance.extra_galaxies)):
            if hasattr(result.instance.extra_galaxies[extra_galaxy_index], "mass"):
                extra_galaxies[extra_galaxy_index].mass.centre = (
                    result.instance.extra_galaxies[extra_galaxy_index].mass.centre
                )
                extra_galaxies[extra_galaxy_index].mass.einstein_radius = (
                    result.model.extra_galaxies[extra_galaxy_index].mass.einstein_radius
                )
                if free_centre:
                    extra_galaxies[extra_galaxy_index].mass.centre = (
                        result.model.extra_galaxies[extra_galaxy_index].mass.centre
                    )

    elif light_as_model:
        extra_galaxies = result.instance.extra_galaxies.as_model((LightProfile,))

        for extra_galaxy_index in range(len(result.instance.extra_galaxies)):
            if extra_galaxies[extra_galaxy_index].bulge is not None:
                extra_galaxies[extra_galaxy_index].bulge.centre = (
                    result.instance.extra_galaxies[extra_galaxy_index].bulge.centre
                )
                if free_centre:
                    extra_galaxies[extra_galaxy_index].bulge.centre = (
                        result.model.extra_galaxies[extra_galaxy_index].bulge.centre
                    )

    else:
        extra_galaxies = result.instance.extra_galaxies.as_model(())

    return extra_galaxies


def lp_chain_tracer_from(light_result, settings_search):
    """
    Output the chain tracer, which is a tracer used for passing the `intensity` values of all linear light profiles
    from the LIGHT PIPELINE to the LIGHT DARK MASS PIPELINE.

    When linear light profiles are used in the LIGHT PIPELINE, their intensities are solved for via linear algebra
    when setting up their corresponding light and mass profiles for the MASS LIGHT DARK PIPELINE.

    This calculation is not numerically accuracy to a small amount (of order 1e-8) in the `intensity` values that
    are solved for. This lack of accuracy will not impact the lens modeling in a noticeable way.

    However, when a pipeline is rerun the `intensity` values that are solved for may change by a small amount,
    changing the unique identifier of the fit, meaning the run does not resume correctly.

    This function outputs the chaining tracer used to pass the `intensity` values to a .json file, or loads
    it from this file if it is already there. This ensures that when a pipeline is rerun, the same `intensity`
    values are always used.
    """

    if settings_search.unique_tag is not None:
        unique_tag = settings_search.unique_tag
    else:
        unique_tag = ""

    lp_chain_tracer_path = (
        Path(conf.instance.output_path)
        / settings_search.path_prefix
        / unique_tag
        / "light_dark_lp_chain_tracer.json"
    )

    try:
        lp_chain_tracer = from_json(lp_chain_tracer_path)

    except FileNotFoundError:
        lp_chain_tracer = (
            light_result.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
        )

        output_to_json(obj=lp_chain_tracer, file_path=lp_chain_tracer_path)

    return lp_chain_tracer


def basis_no_linear_from(name: str, lp_chain_tracer) -> af.Model:
    """
    Returns a basis containing standard light profiles from a basis (e.g. an MGE) in the LIGHT PIPELINE result,
    for the TOTAL MASS PIPELINE.

    For example, if the light pipeline fits a basis of linear Gaussians, this function will return a basis of standard
    Gaussians where each Gaussian has been converted to standard light profile.

    This is used to fix the lens light subtraction in the TOTAL MASS PIPELINE, as opposed to continue using
    linear light profiles which solve for the intensities for each iteration of the pipeline.

    These profiles will have been converted from standard light profiles / linear light profiles to light and mass
    profiles, where their light profile parameters (e.g. their `centre`, `ell_comps`) are used to set up the parameters
    of the light and mass profile.

    If a linear light profile is passed in, the function will return a light and mass profile where its `intensity` is
    fixed to the solved for value of the maximum log likelihood linear fit.

    The light and mass profiles can also be switched to variants which have a radial gradient in their mass-to-light
    conversion, by setting the `include_mass_to_light_gradient` parameter to `True`.


    Parameters
    ----------
    light_result
        The result of the light pipeline, which determines the light and mass profiles used in the LIGHT DARK
        MASS PIPELINE.
    name
        The name of the light profile in the light pipeline's galaxy model that the model is being created for
        (e.g. `bulge`).

    Returns
    -------
    The light and mass profile for a basis (e.g. an MGE) whose priors are initialized from a previous result.
    """

    try:
        lp_instance = getattr(
            lp_chain_tracer.galaxies[0],
            name,
        )
    except AttributeError:
        return None

    if lp_instance is None:
        return None

    profile_list = lp_instance.profile_list

    lp_model_list = []

    for i, light_profile in enumerate(profile_list):
        lp_model = af.Model(lp.Gaussian)

        lp_model.centre = light_profile.centre
        lp_model.ell_comps = light_profile.ell_comps
        lp_model.intensity = light_profile.intensity
        lp_model.sigma = light_profile.sigma

        lp_model_list += [lp_model]

    return af.Model(Basis, profile_list=lp_model_list)


def mass_light_dark_lmp_from(
    light_result: Result,
    name: str,
    lp_chain_tracer,
    linear_lp_to_standard: bool = False,
    light_is_model: bool = False,
    use_gradient: bool = False,
):
    """
    Returns a light and mass profile from a standard light profile (e.g. a Sersic) in the LIGHT PIPELINE result, for
    the LIGHT DARK MASS PIPELINE.

    For example, if the light pipeline fits a Sersic profile, this function will return a Sersic profile for the LIGHT
    DARK MASS PIPELINE.

    For the light profile, it will by default use a linear light and mass profile, which therefore continues to solve
    for the intensity of the linear light profile via linear algebra. This means that during the fit the `intensity`
    used to compute deflection angles is not the same as the `intensity` parameter solved for in the linear light
    profile. If the input `linear_lp_to_standard` is `True`, the function will instead use a standard light profile.

    The light and mass profiles can also be switched to variants which have a radial gradient in their mass-to-light
    conversion, by setting the `include_mass_to_light_gradient` parameter to `True`.

    Parameters
    ----------
    light_result
        The result of the light pipeline, which determines the light and mass profiles used in the LIGHT DARK
        MASS PIPELINE.
    name
        The name of the light profile in the light pipeline's galaxy model that the model is being created for
    light_is_model
        If `True`, the light profile is passed as a model component, else it is a fixed instance.

    Returns
    -------
    The light and mass profile for a standard light profile whose priors are initialized from a previous result.
    """

    lp_instance = getattr(light_result.instance.galaxies.lens, name)
    lp_model = getattr(light_result.model.galaxies.lens, name)

    try:
        is_linear = False

        if not use_gradient:
            lp_to_lmp_dict = {lp.Sersic: lmp.Sersic}
        else:
            lp_to_lmp_dict = {lp.Sersic: lmp.SersicGradient}

        lmp_model = lp_to_lmp_dict[type(lp_instance)]

    except KeyError:
        if not linear_lp_to_standard:
            if not use_gradient:
                lp_linear_to_lmp_dict = {lp_linear.Sersic: lmp_linear.Sersic}
            else:
                lp_linear_to_lmp_dict = {lp_linear.Sersic: lmp_linear.SersicGradient}
        else:
            if not use_gradient:
                lp_linear_to_lmp_dict = {lp_linear.Sersic: lmp.Sersic}
            else:
                lp_linear_to_lmp_dict = {lp_linear.Sersic: lmp.SersicGradient}

        is_linear = True
        lmp_model = lp_linear_to_lmp_dict[type(lp_instance)]

    lmp_model = af.Model(lmp_model)

    if light_is_model:
        lmp_model.take_attributes(source=lp_model)
    else:
        lmp_model.take_attributes(source=lp_instance)

    if is_linear:
        fit = light_result.max_log_likelihood_fit
        lp_solved = getattr(lp_chain_tracer.galaxies[0], name)

        lmp_model.intensity = lp_solved.intensity

    return lmp_model


def mass_light_dark_basis_from(
    light_result: Result,
    name: str,
    lp_chain_tracer,
    linear_lp_to_standard: bool = False,
    use_gradient: bool = False,
) -> af.Model:
    """
    Returns a basis containing light and mass profiles from a basis (e.g. an MGE) in the LIGHT PIPELINE result, for the
    LIGHT DARK MASS PIPELINE.

    For example, if the light pipeline fits a basis of Gaussians, this function will return a basis of Gaussians for the
    LIGHT DARK MASS PIPELINE where each Gaussian has been converted to a light and mass profile.

    These profiles will have been converted from standard light profiles / linear light profiles to light and mass
    profiles, where their light profile parameters (e.g. their `centre`, `ell_comps`) are used to set up the parameters
    of the light and mass profile.

    For the light profile, it will by default use a linear light and mass profile, which therefore continues to solve
    for the intensity of the linear light profile via linear algebra. This means that during the fit the `intensity`
    used to compute deflection angles is not the same as the `intensity` parameter solved for in the linear light
    profile. If the input `linear_lp_to_standard` is `True`, the function will instead use a standard light profile.

    The light and mass profiles can also be switched to variants which have a radial gradient in their mass-to-light
    conversion, by setting the `include_mass_to_light_gradient` parameter to `True`.

    Parameters
    ----------
    light_result
        The result of the light pipeline, which determines the light and mass profiles used in the LIGHT DARK
        MASS PIPELINE.
    name
        The name of the light profile in the light pipeline's galaxy model that the model is being created for
        (e.g. `bulge`).

    Returns
    -------
    The light and mass profile for a basis (e.g. an MGE) whose priors are initialized from a previous result.
    """

    try:
        lp_instance = getattr(
            lp_chain_tracer.galaxies[0],
            name,
        )
    except AttributeError:
        return None

    profile_list = lp_instance.profile_list

    lmp_model_list = []

    for i, light_profile in enumerate(profile_list):
        if not linear_lp_to_standard:
            if not use_gradient:
                lmp_model = af.Model(lmp_linear.Gaussian)
            else:
                lmp_model = af.Model(lmp_linear.GaussianGradient)
        else:
            if not use_gradient:
                lmp_model = af.Model(lmp.Gaussian)
            else:
                lmp_model = af.Model(lmp.GaussianGradient)

        lmp_model.centre = light_profile.centre
        lmp_model.ell_comps = light_profile.ell_comps

        lmp_model.intensity = float(light_profile.intensity)
        lmp_model.sigma = light_profile.sigma

        lmp_model_list += [lmp_model]

        if not use_gradient:
            lmp_model.mass_to_light_ratio = lmp_model_list[0].mass_to_light_ratio
        else:
            lmp_model.mass_to_light_ratio_base = lmp_model_list[
                0
            ].mass_to_light_ratio_base
            lmp_model.mass_to_light_gradient = lmp_model_list[0].mass_to_light_gradient

    return af.Model(Basis, profile_list=lmp_model_list)


def mass_light_dark_from(
    light_result: Result,
    name: str,
    lp_chain_tracer,
    linear_lp_to_standard: bool = False,
    light_is_model: bool = False,
    use_gradient: bool = False,
) -> Optional[af.Model]:
    """
    Returns light and mass profiles from the LIGHT PIPELINE result, for the LIGHT DARK MASS PIPELINE.

    For example, if the light pipeline fits Sersic profiles, this function will return Sersic profiles for the LIGHT
    DARK MASS PIPELINE.

    These profiles will have been converted from standard light profiles / linear light profiles to light and mass
    profiles, where their light profile parameters (e.g. their `centre`, `ell_comps`) are used to set up the parameters
    of the light and mass profile.

    If a linear light profile is passed in, the function will return a mass profile where its `intensity` is
    fixed to the solved for value of the maximum log likelihood linear fit.

    For the light profile, it will by default use a linear light and mass profile, which therefore continues to solve
    for the intensity of the linear light profile via linear algebra. This means that during the fit the `intensity`
    used to compute deflection angles is not the same as the `intensity` parameter solved for in the linear light
    profile. If the input `linear_lp_to_standard` is `True`, the function will instead use a standard light profile.

    This function supports the input of a basis (e.g. an MGE), converting every individual standard light or
    linear light profile in the basis to a light and mass profile.

    The light and mass profiles can also be switched to variants which have a radial gradient in their mass-to-light
    conversion, by setting the `include_mass_to_light_gradient` parameter to `True`.


    Parameters
    ----------
    light_result
        The result of the light pipeline, which determines the light and mass profiles used in the LIGHT DARK
        MASS PIPELINE.
    name
        The name of the light profile in the light pipeline's galaxy model that the model is being created for
        (e.g. `bulge`).
    linear_lp_to_standard
        If `True`, the light profile is passed as a standard component, else it is a linear component. The
        mass always uses the `interity` parameter of the linear light profile solved for in the previous pipeline.
    light_is_model
        If `True`, the light profile is passed as a model component, else it is a fixed instance. For a basis
        (e.g. an MGE) this feature is not used due to the large number of profiles in the basis.
    use_gradient
        If `True`, the light and mass profiles is switched to a variant which includes additional parameters that
        model a radial gradient in the mass-to-light ratio.

    Returns
    -------
    The light and mass profile whose priors are initialized from a previous result.
    """

    lp_instance = getattr(light_result.instance.galaxies.lens, name)

    if lp_instance is None:
        return None

    if not isinstance(lp_instance, Basis):
        return mass_light_dark_lmp_from(
            light_result=light_result,
            name=name,
            lp_chain_tracer=lp_chain_tracer,
            linear_lp_to_standard=linear_lp_to_standard,
            light_is_model=light_is_model,
            use_gradient=use_gradient,
        )
    return mass_light_dark_basis_from(
        light_result=light_result,
        lp_chain_tracer=lp_chain_tracer,
        name=name,
        use_gradient=use_gradient,
    )


def link_ratios(link_mass_to_light_ratios: bool, light_result, bulge, disk):
    """
    Links the mass to light ratios and gradients of the bulge and disk profiles in the MASS LIGHT DARK PIPELINE.

    The following use cases are supported:

    1) Link the mass to light ratios of the bulge and diskwhen they are both sersic light and mass profile.
    2) Link the mass to light ratios of the bulge and disk when for a basis (e.g. an MGE).
    3) Does approrpiate linking with the gradient of the mass-to-light ratio for the basis profiles.

    Parameters
    ----------
    link_mass_to_light_ratios
        Whether the mass-to-light ratios of the bulge and disk profiles are linked.
    light_result
        The result of the light pipeline, which determines the light and mass profiles used in the MASS LIGHT DARK
        PIPELINE.
    bulge
        The bulge model light profile.
    disk
        The disk model light profile.

    Returns
    -------
    The bulge and disk profiles with the mass-to-light ratios linked.
    """

    if bulge is None or disk is None:
        return bulge, disk

    if not link_mass_to_light_ratios:
        return bulge, disk

    bulge_instance = getattr(light_result.instance.galaxies.lens, "bulge")

    if not isinstance(bulge_instance, Basis):
        bulge.mass_to_light_ratio = disk.mass_to_light_ratio

        return bulge, disk

    for bulge_lp in bulge.profile_list:
        try:
            bulge_lp.mass_to_light_ratio = disk.profile_list[0].mass_to_light_ratio
        except AttributeError:
            bulge_lp.mass_to_light_ratio_base = disk.profile_list[
                0
            ].mass_to_light_ratio_base
            bulge_lp.mass_to_light_gradient = disk.profile_list[
                0
            ].mass_to_light_gradient

    return bulge, disk
