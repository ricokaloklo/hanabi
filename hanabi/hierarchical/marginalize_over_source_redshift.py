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
        zs = np.linspace(0., self.population_parameter_dict["redshift_max"], num=100000)
        out = scipy.integrate.simps(self.prob(pd.DataFrame({'redshift': zs})), zs)
        return out

class Marginalization(object):
    def __init__(
        self,
        result_file,
        mass_src_model=Marginalized(),
        spin_src_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity()
    ):
        self.result = bilby.result.read_in_result(result_file)
        self.mass_src_model = mass_src_model
        self.spin_src_model = spin_src_model
        self.merger_rate_density_src_pop_model = merger_rate_density_src_pop_model

    def marginalize_over_params(self):
        pass

class MarginalizationWithMagnification(Marginalization):
    def marginalize_over_abs_magnification(self):
        pass