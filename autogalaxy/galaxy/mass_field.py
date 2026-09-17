"""
The `MassField` class is the redshift-bearing container for the mass components which describe the tidal field
of everything *outside* the system being modelled.

A strong lens model does not only contain the mass of the galaxies being fitted: line-of-sight structure and the
group or cluster environment the lens sits in also deflect light. These contributions are described by mass
components with no light of their own — `ExternalShear`, `MassSheet` and `ExternalPotential` are the intended
ones, though any `MassProfile` is accepted (e.g. an NFW "environment" halo standing in for a nearby group).

A `MassField` is its own thing, not a `Galaxy` subclass. It contributes nothing to light and everything to mass,
so a `Galaxies` plane can hold one beside its galaxies and every aggregate mass calculation (deflections,
convergence, potential) picks it up while every light calculation sees zeros.

The galaxy-attached form remains fully supported and is not deprecated: writing
`Galaxy(redshift=0.5, shear=ExternalShear(gamma_1=0.05, gamma_2=0.05))` behaves exactly as it always has. The
`MassField` simply gives the external field a home of its own, so it need not be bolted onto a galaxy whose
light and mass describe something physically different.

In **PyAutoLens** the consumer is `Tracer(fields=...)`, which places each `MassField` at its own redshift in the
multi-plane ray-tracing calculation. That is phase 2 of this work and lives in PyAutoLens, not in this repo.
"""

from typing import Dict, List, Optional

import numpy as np

from autonerves.dictable import instance_as_dict, to_dict

import autoarray as aa
import autofit as af

from autogalaxy import exc
from autogalaxy.galaxy.mass_aggregate import MassProfileAggregate
from autogalaxy.profiles.geometry_profiles import GeometryProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles import validate


class MassField(af.ModelObject, MassProfileAggregate):
    """
    The mass components describing the tidal field of everything outside the modelled system, at a redshift.

    A `MassField` holds its components as named keyword-argument attributes, exactly like a `Galaxy`, so the
    user accesses them by name (e.g. `field.shear`, `field.mass_sheet`). Only `MassProfile` objects are
    accepted: light profiles, pixelizations and regularizations belong on a `Galaxy`.

    The intended components are the ones which describe external structure rather than a galaxy —
    `ExternalShear`, `MassSheet` and `ExternalPotential` — but any `MassProfile` may be used, for example an
    NFW halo representing the group environment.

    Because it has no light, its image is zeros everywhere and its light-profile list is empty. A
    `Galaxies([galaxy, field])` therefore sums mass contributions from both and light from the galaxy alone.

    @DynamicAttrs
    """

    def __init__(self, redshift: float, **kwargs):
        """
        Class representing the external mass field at a given redshift.

        Parameters
        ----------
        redshift
            The redshift of the external field.
        kwargs
            The named mass components of the field (e.g. `shear`, `mass_sheet`, `potential`), each of which
            must be a `MassProfile`.
        """
        # `label` is an `af.ModelObject` attribute, not a component of the field. It rides in the `arguments`
        # of `dict()` and therefore comes back as a keyword argument on a `from_dict` round-trip, so it is
        # handed to `ModelObject` rather than validated as a mass profile.
        super().__init__(label=kwargs.pop("label", None))

        validate.validate_redshift(redshift=redshift)

        self.redshift = redshift

        for name, val in kwargs.items():
            if isinstance(val, list):
                raise exc.GalaxyException(
                    "One or more of the input mass profiles has been passed to the MassField object"
                    "as a list."
                    ""
                    "The MassField object cannot accept a list of mass profiles. "
                    ""
                    "Instead, pass these objects as a dictionary, where the key of each dictionary entry is"
                    "the name of the profile and the value is the profile, e.g.:"
                    ""
                    "{shear : al.mp.ExternalShear()}"
                    ""
                )

            if not isinstance(val, MassProfile):
                raise exc.GalaxyException(
                    f"MassField received '{name}' = {type(val).__name__}, which is not a MassProfile. A "
                    f"MassField holds only the mass components describing the tidal field of everything "
                    f"outside the modelled system (ExternalShear, MassSheet, ExternalPotential, or any other "
                    f"MassProfile); put light profiles, pixelizations and regularizations on a Galaxy."
                )

            setattr(self, name, val)

    def __hash__(self):
        return int(self.id)

    def __repr__(self):
        profile_names = ", ".join(self.profile_dict.keys())

        if profile_names:
            return f"MassField(redshift={self.redshift}, {profile_names})"

        return f"MassField(redshift={self.redshift})"

    def __eq__(self, other):
        return self.dict() == other.dict()

    @property
    def profile_dict(self) -> Dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, GeometryProfile)
        }

    def dict(self) -> Dict:
        d = instance_as_dict(self)
        d["arguments"] = {
            **d.get("arguments", {}),
            **{name: to_dict(profile) for name, profile in self.profile_dict.items()},
        }
        return d

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        """
        Returns an empty list, because a `MassField` holds no light profiles.

        The `Galaxies` class builds its list of images by calling this on each member, so the field simply
        contributes nothing to it.
        """
        return []

    def image_2d_list_unbinned_from(
        self, grid: aa.Grid2D, xp=np, operated_only: Optional[bool] = None
    ) -> List[np.ndarray]:
        """Returns an empty list: a `MassField` holds no light profiles to evaluate before binning."""
        return []

    @aa.decorators.to_array
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns a 2D image of zeros, because a `MassField` has no light.

        This mirrors the zeros returned by `Galaxy.image_2d_from` for a galaxy with no light profiles, so the
        sum performed by `Galaxies.image_2d_from` is unaffected by the presence of a field.
        """
        return xp.zeros((grid.shape[0],))

    def image_2d_unbinned_from(
        self, grid: aa.Grid2D, xp=np, operated_only: Optional[bool] = None
    ) -> np.ndarray:
        """Returns zeros on the over-sampled grid, mirroring the light-less path of `Galaxy`."""
        return xp.zeros((grid.over_sampled.shape[0],))
