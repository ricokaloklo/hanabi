import numpy as np
import pandas as pd
import scipy.integrate
import bilby
from .source_population_model import SourcePopulationModel, Marginalized
from .merger_rate_density import MergerRateDensity, MarginalizedMergerRateDensity
from ..lensing.optical_depth import *
from ..lensing.absolute_magnification import *

class LensedSourceRedshiftProbDist(SourcePopulationModel):
    def __init__(self, merger_rate_density, optical_depth, redshift_max):
        super(LensedSourceRedshiftProbDist, self).__init__(
            signal_parameter_names=['redshift'],
            population_parameter_dict={
                'redshift_max': redshift_max
            }
        )
        self.merger_rate_density = merger_rate_density
        self.optical_depth = optical_depth

        self.normalization = 1.0
        self.normalization = self.compute_normalization()

    def prob(self, dataset):
        return (1.0/self.normalization) * self.optical_depth.evaluate(dataset["redshift"]) * self.merger_rate_density.prob(dataset)

    def compute_normalization(self):
        zs = np.arange(0., self.population_parameter_dict["redshift_max"], step=0.1)
        out = scipy.integrate.simps(self.prob(pd.DataFrame({'redshift': zs})), zs)
        return out

class NotLensedSourceRedshiftProbDist(LensedSourceRedshiftProbDist):
    def prob(self, dataset):
        return (1.0/self.normalization) * (1.0 - self.optical_depth.evaluate(dataset["redshift"])) * self.merger_rate_density.prob(dataset)
