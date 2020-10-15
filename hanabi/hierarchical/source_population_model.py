import numpy as np
import bilby.gw.conversion
import gwpopulation.models

class SourcePopulationPrior(object):
    def __init__(self, signal_parameter_names, population_parameter_dict):
        self.signal_parameter_names = signal_parameter_names
        self.population_parameter_dict = population_parameter_dict

    @staticmethod
    def _check_if_keys_exist(names, keys):
        if all([name in keys for name in names]):
            return True
        else:
            return False

    def _parameter_conversion(self, dataset):
        # Implement a parameter conversion function
        return dataset

    def prob(self, dataset):
        if not self._check_if_keys_exist(names=self.signal_parameter_names, keys=dataset.keys()):
            # Parameters needed for evaluation does not exist
            self._parameter_conversion(dataset)

        return self._prob(dataset)

    def _prob(self, dataset):
        raise NotImplementedError

    def ln_prob(self, dataset):
        return np.log(self.prob(dataset))


class Marginalized(SourcePopulationPrior):
    def __init__(self):
        super(Marginalized, self).__init__([], {})

    def _check_if_keys_exist(names, keys):
        return True

    def prob(self, dataset):
        return 1.0


class MarginalizedMergerRateDensity(Marginalized):
    def total_number_of_mergers(self, T_obs):
        return 1.0
 

# Wrapper for gwpopulation's power_law_primary_mass_ratio
class PowerLawPrimaryMassRatio(SourcePopulationPrior):
    def __init__(self, alpha, beta, mmin, mmax):
        super(PowerLawPrimaryMassRatio, self).__init__(
            signal_parameter_names=[
                'mass_1_source',
                'mass_ratio'
            ],
            population_parameter_dict={
                'alpha': alpha,
                'beta': beta,
                'mmin': mmin,
                'mmax': mmax
            }
        )

    def _parameter_conversion(self, dataset):
        if self._check_if_keys_exist(names=["mass_1_source", "mass_2_source"], keys=dataset.keys()):
            # Convert mass_1, mass_2 to mass_1, mass_ratio
            mass_ratio = bilby.gw.conversion.component_masses_to_mass_ratio(
                mass_1=dataset["mass_1_source"],
                mass_2=dataset["mass_2_source"]
            )
            dataset["mass_ratio"] = mass_ratio

    def _prob(self, dataset):
        return gwpopulation.models.mass.power_law_primary_mass_ratio(
            {"mass_1": dataset["mass_1_source"], "mass_ratio": dataset["mass_ratio"]},
            self.population_parameter_dict["alpha"],
            self.population_parameter_dict["beta"],
            self.population_parameter_dict["mmin"],
            self.population_parameter_dict["mmax"]
        )

# Wrapper for gwpopulation's PowerLawRedshift
class PowerLawRedshift(SourcePopulationPrior):
    def __init__(self, R_0, kappa, redshift_max=2.3):
        super(PowerLawRedshift, self).__init__(
            signal_parameter_names=[
                "redshift"
            ],
            population_parameter_dict={
                'R_0': R_0,
                'kappa': kappa,
                'redshift_max': redshift_max
            }
        )

        self._PowerLawRedshift = gwpopulation.models.redshift.PowerLawRedshift(
            z_max=self.population_parameter_dict["redshift_max"]
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

    def _prob(self, dataset):
        return np.log(self.population_parameter_dict["R_0"]) + self._PowerLawRedshift.probability(
            dataset=dataset, lamb=self.population_parameter_dict["kappa"]
        )

    def total_number_of_mergers(self, T_obs):
        # R_0 is in the unit of Gpc^-3 T_obs^-1
        # Since astropy.cosmology is using Mpc as the default unit
        return self.population_parameter_dict["R_0"] / 1e9 * T_obs * \
            self._PowerLawRedshift.normalisation(parameters={'lamb': self.population_parameter_dict["kappa"]})
