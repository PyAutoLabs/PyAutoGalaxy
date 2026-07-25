from typing import List

from autogalaxy.analysis.result import ResultDataset
from autogalaxy.ellipse.ellipse.ellipse import Ellipse
from autogalaxy.ellipse.fit_ellipse import FitEllipse


class ResultEllipse(ResultDataset):
    @property
    def max_log_likelihood_fit_list(self) -> FitEllipse:
        """
        An instance of a `FitEllipse` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """
        return self.analysis.fit_list_from(instance=self.instance)

    @property
    def max_log_likelihood_ellipses(self) -> List[Ellipse]:
        """
        An instance of galaxies corresponding to the maximum log likelihood model inferred by the non-linear search.

        The galaxies list is computed from the `max_log_likelihood_fit`, as this ensures that all linear light profiles
        are converted to normal light profiles with their `intensity` values updated.
        """
        return self.instance.ellipses
