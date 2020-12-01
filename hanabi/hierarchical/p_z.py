import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import bilby
from .source_population_model import SourcePopulationModel, Marginalized
from .merger_rate_density import MergerRateDensity, MarginalizedMergerRateDensity
from ..lensing.optical_depth import *
from ..lensing.absolute_magnification import *

class LensedSourceRedshiftProbDist(SourcePopulationModel):
    def __init__(self, merger_rate_density, optical_depth):
        super(LensedSourceRedshiftProbDist, self).__init__(
            signal_parameter_names=['redshift'],
            population_parameter_dict={
                'redshift_max': merger_rate_density.population_parameter_dict["redshift_max"]
            }
        )
        self.merger_rate_density = merger_rate_density
        self.optical_depth = optical_depth

        self.normalization = 1.0
        self.normalization = self.compute_normalization()

    def _prob(self, dataset):
        return (1.0/self.normalization) * self.optical_depth.evaluate(dataset["redshift"]) * self.merger_rate_density.prob(dataset)

    def sample(self):
        # Draw sample using empirical supremum rejection sampling
        out = None
        c_hat = 10 # Initial guess for the supremum
        while out is None:
            proposed_pt = np.random.uniform(0., self.population_parameter_dict["redshift_max"])
            u = np.random.random()
            pdf_ev = self._prob({'redshift': proposed_pt})
            if u <= pdf_ev/(c_hat/self.population_parameter_dict["redshift_max"]):
                out = proposed_pt
            c_hat = max(c_hat, pdf_ev/(1./self.population_parameter_dict["redshift_max"]))

        return out

    def cdf(self, z_upper):
        zs = np.linspace(0., z_upper, num=1000)
        out = scipy.integrate.simps(self.prob({'redshift': zs}), zs)
        return out

    def compute_normalization(self):
        return self.cdf(self.population_parameter_dict["redshift_max"])

class NotLensedSourceRedshiftProbDist(LensedSourceRedshiftProbDist):
    def _prob(self, dataset):
        return (1.0/self.normalization) * (1.0 - self.optical_depth.evaluate(dataset["redshift"])) * self.merger_rate_density.prob(dataset)
