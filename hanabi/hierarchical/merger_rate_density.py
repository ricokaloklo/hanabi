import numpy as np
import bilby.gw.conversion
import gwpopulation
from .source_population_model import SourcePopulationModel

class MergerRateDensity(SourcePopulationModel):
    def __init__(self, population_parameter_dict):
        super(MergerRateDensity, self).__init__(
            signal_parameter_names=["redshift"],
            population_parameter_dict=population_parameter_dict
        )
    
    def _parameter_conversion(self, dataset):
        if "luminosity_distance" in dataset.keys():
            # Calculate redshift from luminosity distance
            redshift = bilby.gw.conversion.luminosity_distance_to_redshift(
                dataset["luminosity_distance"]
            )
        elif "comoving_distance" in dataset.keys():
            # Calculate redshift from comoving distance
            redshift = bilby.gw.conversion.comoving_distance_to_redshift(
                dataset["comoving_distance"]
            )
        else:
            raise ValueError("No distance measure in dataset")

        dataset["redshift"] = redshift

# Wrapper for gwpopulation's PowerLawRedshift
class PowerLawRedshift(MergerRateDensity):
    def __init__(self, R_0, kappa, redshift_max=2.3):
        super(PowerLawRedshift, self).__init__(
            population_parameter_dict={
                'R_0': R_0,
                'kappa': kappa,
                'redshift_max': redshift_max
            }
        )

        self._PowerLawRedshift = gwpopulation.models.redshift.PowerLawRedshift(
            z_max=self.population_parameter_dict["redshift_max"]
        )

    def _prob(self, dataset):
        return self._PowerLawRedshift.probability(
            dataset=dataset, lamb=self.population_parameter_dict["kappa"]
        )

    def ln_dN_over_dz(self, dataset):
        # NOTE This is not a normalized probability
        return np.log(self.population_parameter_dict["R_0"] / 1e9) + np.log(
            self._PowerLawRedshift.differential_spacetime_volume(
                dataset=dataset, lamb=self.population_parameter_dict["kappa"]
            )
        )

    def total_number_of_mergers(self, T_obs):
        # R_0 is in the unit of Gpc^-3 T_obs^-1
        # Since astropy.cosmology is using Mpc as the default unit
        return self.population_parameter_dict["R_0"] / 1e9 * T_obs * \
            self._PowerLawRedshift.normalisation(parameters={'lamb': self.population_parameter_dict["kappa"]})

class AnalyticalBBHMergerRateDensity(MergerRateDensity):
    def __init__(self, a_1, a_2, a_3, a_4, redshift_max):
        super(AnalyticalBBHMergerRateDensity, self).__init__(
            population_parameter_dict={
                'a_1': a_1,
                'a_2': a_2,
                'a_3': a_3,
                'a_4': a_4,
                'redshift_max': redshift_max,
            }
        )

        # The merger rate density fit has a unit of Gpc^-3 yr^-1
        self.population_parameter_dict["R_0"] = self.evaluate_fit(0.0)

    def evaluate_fit(self, z):
        a_1 = self.population_parameter_dict['a_1']
        a_2 = self.population_parameter_dict['a_2']
        a_3 = self.population_parameter_dict['a_3']
        a_4 = self.population_parameter_dict['a_4']
        redshift_max = self.population_parameter_dict['redshift_max']

        if 0. <= z < redshift_max:
            return a_1*np.exp(a_2*z)/(a_4 + np.exp(a_3*z))
        else:
            return 0.0

class BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    def __init__(self):
        super(BelczynskiEtAl2017PopIPopIIStars, self).__init__(
            6.6e3,
            1.6,
            2.1,
            30,
            15
        )

class BelczynskiEtAl2017PopIIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    def __init__(self):
        super(BelczynskiEtAl2017PopIIIStars, self).__init__(
            6e4,
            1.0,
            1.4,
            3e6,
            45
        )

class KinugawaEtAl2016PopIIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    def __init__(self):
        super(KinugawaEtAl2016PopIIIStars, self).__init__(
            1e4,
            0.7,
            1.1,
            500,
            45
        )   