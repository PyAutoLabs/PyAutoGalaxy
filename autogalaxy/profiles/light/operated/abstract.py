class LightProfileOperated:
    """
    Mixin class that marks a light profile as already having had an instrument operation applied to it.

    An "operated" light profile represents emission whose image has already had an operation applied, most
    commonly a PSF convolution. This means that when the image of an operated light profile is computed, the
    PSF convolution step is skipped — the PSF effect is already baked into the profile itself.

    This pattern is useful for modelling point-source emission (e.g. AGN) or other compact emission where the
    PSF profile itself is used directly as the light profile.

    The `operated_only` input to `image_2d_from` methods throughout the codebase controls which light profiles
    contribute to an image:

    - `operated_only=None` (default): all light profiles contribute regardless of whether they are operated.
    - `operated_only=True`: only `LightProfileOperated` instances contribute; non-operated profiles return zeros.
    - `operated_only=False`: only non-operated profiles contribute; `LightProfileOperated` instances return zeros.
    """

    pass
