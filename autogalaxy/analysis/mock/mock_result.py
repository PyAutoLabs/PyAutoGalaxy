from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from autogalaxy import mock

import autofit as af
import autogalaxy as ag


class MockResult(af.m.MockResult):
    def __init__(
        self,
        samples: mock.MockSamples = None,
        samples_summary: af.m.MockSamplesSummary = None,
        instance: af.Instance = None,
        model: af.Model = None,
        analysis: mock.MockAnalysis = None,
        search: af.mock.MockSearch = None,
        max_log_likelihood_galaxies: List[ag.Galaxy] = None,
        max_log_likelihood_tracer=None,
    ):
        super().__init__(
            samples=samples,
            samples_summary=samples_summary,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.max_log_likelihood_galaxies = max_log_likelihood_galaxies
        self.max_log_likelihood_tracer = max_log_likelihood_tracer

    @property
    def last(self):
        return self
