# Linear-operated light-profile priors

`linear_operated.Sersic` intentionally inherits its priors from
`light/linear/sersic.yaml` through its `linear.Sersic` base class. There is no
separate `linear_operated/sersic.yaml`: the operated profile uses the same
centre, ellipticity, effective-radius and Sérsic-index priors. Its intensity
is solved linearly rather than assigned a non-linear prior.

Keep changes to these shared defaults in `light/linear/sersic.yaml`. The
inheritance is checked in `test_autogalaxy/test_priors_config.py`.
