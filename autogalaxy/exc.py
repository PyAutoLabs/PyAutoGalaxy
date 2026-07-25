import autofit as af
from autofit.exc import *
from autoarray.exc import *


class ProfileException(Exception):
    """
    Raises exceptions associated with the `profile` modules and `LightProfile` / `MassProfile` classes.

    For example when a mass-profile deflection angle calculation goes wrong due to numerical issues.
    """

    pass


class GalaxyException(Exception):
    """
    Raises exceptions associated with the `galaxy` module and `Galaxy` class.

    For example if stellar-mass specific calculations are used for a galaxy which does not have a stellar mas
    component.
    """

    pass


class AnalysisException(Exception):
    """
    Raises exceptions associated with the `analysis` modules in the `model` packages and `Analysis` classes.

    For example if the figure of merit of the analysis class's `log_likelihood_function` has changed for a resumed
    run from a previous run.
    """

    pass


class PixelizationException(af.exc.FitException):
    """
    Raises exceptions associated with the `inversion/pixelization` modules and `Pixelization` classes.

    For example if a `RectangularAdaptDensity` mesh has dimensions below 3x3.

    This exception overwrites `autoarray.exc.PixelizationException` in order to add a `FitException`. This means that
    if this exception is raised during a model-fit in the analysis class's `log_likelihood_function` that model
    is resampled and does not terminate the code.
    """

    pass


class UnitsException(Exception):
    """
    Raises exceptions associated with unit conversions.

    For example if when constructing a dark matter profile the units and format of the redshifts are input
    incorrectly.
    """

    pass


def raise_linear_light_profile_in_unmasked():
    raise FitException(
        """
        You cannot compute the unmasked image from a fit which contains linear 
        light profiles or a pixelization, and therefore uses an inversion.
        
        This is because the inversion can only be performed within the masked
        region.
        """
    )


def raise_linear_light_profile_in_plot(
    plotter_type: str,
):
    raise ProfileException(
        f"""
        A linear light profile has been passed to the `{plotter_type}`.

        Linear light profiles cannot be plotted, because they do not have an
        intensity value.

        Therefore convert all linear light profiles to normal light profiles
        with intensity values. 

        The easiest way to do this is to create a`FitImaging` 
        or `FitInterferometer` object. This will contain a property 
        where all linear light have been converted to regular light profiles 
        using the solved for intensities. 
        
        If you are using PyAutoLens, you should use the attribute
        `fit.tracer_linear_light_profiles_to_light_profiles` to access a 
        `Tracer` with these converted light profiles.

        If you are using PyAutoGalaxy, you should instead use
        `fit.galaxies_linear_light_profiles_to_light_profiles` to access
        `Galaxy` objects with these converted light profiles.
             
        If you are using database functionality and creating tracers
        via the `TracerAgg` object or `GalaxiesAgg` object, you should 
        instead use the `FitImagingAgg` object to create `FitImaging` 
        objects.
        
        You should then access what you need via 
        `fit.tracer_linear_light_profiles_to_light_profiles` or
        `fit.galaxies_linear_light_profiles_to_light_profiles`.
        """
    )


def invalid_linear_algebra_for_visualization_message():
    return (
        "Visualization of the inversion's reconstruction cannot be performed because the "
        "inversion failed to calculate a valid solution (e.g. the linear algebra did not "
        "find a valid solution). \n\n"
        "This should not happen after a search has completed, so if this is seen in the final "
        "results, please contact the developers via the PyAutoLens GitHub or SLACK."
    )
