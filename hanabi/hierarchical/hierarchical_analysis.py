import sys
import numpy as np
import bilby
from .marginalized_likelihood import MonteCarloMarginalizedLikelihood
from .p_z import LensedSourceRedshiftProbDist, NotLensedSourceRedshiftProbDist
from .source_population_model import *
from .merger_rate_density import *
from ..lensing.optical_depth import *
from ..lensing.absolute_magnification import *
from .gibbs_sampling import CustomCollapsedBlockedGibbsSampler
from .parser import create_hierarchical_analysis_parser
from .utils import setup_logger

from .utils import get_version_information
__version__ = get_version_information()
__prog__ = "hanabi_hierarchical_analysis"

class HierarchicalAnalysisInput(object):
    def __init__(self, args, unknown_args):
        self.outdir = args.outdir
        self.request_cpus = args.request_cpus

        self.n_triggers = args.n_triggers
        self.inference_result = args.inference_result

        self.source_population_model = args.source_population_model
        self.optical_depth = args.optical_depth
        self.absolute_magnifications = args.absolute_magnifications
        self.redshift_prior = args.redshift_prior

        self.sampler_kwargs = args.sampler_kwargs
        self.sampling_seed = args.sampling_seed

    @property
    def inference_result(self):
        return self._inference_result

    @inference_result.setter
    def inference_result(self, inference_result):
        if isinstance(inference_result, str):
            self._inference_result = bilby.result.read_in_result(inference_result)
        else:
            self._inference_result = inference_result

    @property
    def sampler_kwargs(self):
        return self._sampler_kwargs

    @sampler_kwargs.setter
    def sampler_kwargs(self, sampler_kwargs):
        self._sampler_kwargs = {
            "nlive": 1000,
            "nact": 20,
            "dlogz": 0.1,
            "sample": "unif",
        }

        if self.request_cpus is int and self.request_cpus > 1:
            self._sampler_kwargs["npool"] = self.request_cpus

        if sampler_kwargs is not None:
            self._sampler_kwargs.update(sampler_kwargs)

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = 1234
        self._sampling_seed = sampling_seed

    def get_likelihood_and_priors(self):
        pass

    def marginalize_over_redshift(self):
        likelihood, priors = self.get_likelihood_and_priors()

        self.marginalization_result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            outdir=self.outdir,
            **self.sampler_kwargs
        )

def main():
    sys.exit(0)