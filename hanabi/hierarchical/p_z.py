import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import bilby
from bilby.core.prior import Interped
from .merger_rate_density import MergerRateDensity, MarginalizedMergerRateDensity
from ..lensing.optical_depth import *
from ..lensing.absolute_magnification import *

class LensedSourceRedshiftProbDist(Interped):
    def __init__(self, merger_rate_density, optical_depth, name=None, latex_label=None):
        self.redshift_max = merger_rate_density.population_parameter_dict["redshift_max"]
        self.merger_rate_density = merger_rate_density
        self.optical_depth = optical_depth

        self.zs = np.linspace(0., self.redshift_max, num=1000)
        self.compute_normalization()
        self.pdfs = self._prob(self.zs)

        super(LensedSourceRedshiftProbDist, self).__init__(
            self.zs,
            self.pdfs,
            minimum=0.,
            maximum=self.redshift_max,
            name=name,
            latex_label=latex_label,
        )

    def to_json(self):
        # Return the (reconstructed) interpolated prior
        interped_prior = Interped(
            self.zs,
            self.pdfs,
            minimum=0.,
            maximum=self.redshift_max,
            name=self.name,
            latex_label=self.latex_label,
        )
        return interped_prior.to_json()

    def _prob(self, z):
        return (1.0/self.normalization) * self.optical_depth.evaluate(z) * self.merger_rate_density.prob({"redshift": z})

    def compute_normalization(self):
        self.normalization = 1.0
        norm = scipy.integrate.simps(self._prob(self.zs), self.zs)
        self.normalization = norm

class NotLensedSourceRedshiftProbDist(LensedSourceRedshiftProbDist):
    def _prob(self, z):
        return (1.0/self.normalization) * (1.0 - self.optical_depth.evaluate(z)) * self.merger_rate_density.prob({'redshift': z})
