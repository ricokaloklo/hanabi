import numpy as np
import bilby.gw.conversion
import gwpopulation.models

class SourcePopulationPrior(object):
    def __init__(self, signal_parameter_names, population_parameter_dict):
        self.signal_parameter_names = signal_parameter_names
        self.population_parameter_dict = population_parameter_dict

    def _check_if_keys_exist(self, dataset):
        if all([name in dataset.keys() for name in self.signal_parameter_names]):
            return True
        else:
            return False

    def _parameter_conversion(self, dataset):
        # Implement a parameter conversion function
        return dataset

    def prob(self, dataset):
        if not self._check_if_keys_exist(dataset):
            # Parameters needed for evaluation does not exist
            self._parameter_conversion(dataset)

        return self._prob(dataset)

    def _prob(self, dataset):
        raise NotImplementedError

    def ln_prob(self, dataset):
        return np.log(self.prob(dataset))

# Wrappers for gwpopulation's power_law_primary_mass_ratio
class PowerLawPrimaryMassRatio(SourcePopulationPrior):
    def __init__(self, alpha, beta, mmin, mmax):
        super(PowerLawPrimaryMassRatio, self).__init__(
            signal_parameter_names=[
                'mass_1',
                'mass_ratio'
            ],
            population_parameter_dict={
                'alpha': alpha,
                'beta': beta,
                'mmin': mmin,
                'mmax': mmax
            }
        )

    def _prob(self, dataset):
        return gwpopulation.models.mass.power_law_primary_mass_ratio(
            dataset,
            self.population_parameter_dict["alpha"],
            self.population_parameter_dict["beta"],
            self.population_parameter_dict["mmin"],
            self.population_parameter_dict["mmax"]
        )

# Wrapper for gwpopulation's PowerLawRedshift
class PowerLawRedshift(SourcePopulationPrior):
    def __init__(self, kappa, redshift_max=2.3):
        super(PowerLawRedshift, self).__init__(
            signal_parameter_names=[
                "redshift"
            ],
            population_parameter_dict={
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
            # Don't know do what (Ng, 2019)
            raise ValueError("No distance measure in dataset")

        dataset["redshift"] = redshift

    def _prob(self, dataset):
        return self._PowerLawRedshift.probability(
            dataset=dataset, lamb=self.population_parameter_dict["kappa"]
        )

    def total_number_of_expected_mergers(self, R0, T_obs):
        # R0 is in the unit of Gpc^-3 T_obs^-1
        # Since astropy.cosmology is using Mpc as the default unit
        return R0 / 1e9 * T_obs * \
            self._PowerLawRedshift.normalisation(parameters={'lamb': self.population_parameter_dict["kappa"]})
