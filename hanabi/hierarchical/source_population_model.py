import numpy as np
import bilby.gw.conversion
import gwpopulation.models

class SourcePopulationModel(object):
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
        # NOTE pandas dataframe is edited *in-place*
        pass

    def prob(self, dataset):
        if not self._check_if_keys_exist(names=self.signal_parameter_names, keys=dataset.keys()):
            # Parameters needed for evaluation does not exist
            self._parameter_conversion(dataset)

        return self._prob(dataset).to_numpy().flatten()

    def _prob(self, dataset):
        raise NotImplementedError

    def ln_prob(self, dataset):
        return np.log(self.prob(dataset))


class Marginalized(SourcePopulationModel):
    def __init__(self):
        super(Marginalized, self).__init__([], {})

    def _check_if_keys_exist(names, keys):
        return True

    def prob(self, dataset):
        return 1.0
 

# Wrapper for gwpopulation's power_law_primary_mass_ratio
class PowerLawPrimaryMassRatio(SourcePopulationModel):
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

class UniformAlignedSpinComponent(SourcePopulationModel):
    def __init__(self):
        super(UniformAlignedSpinComponent, self).__init__(
            signal_parameter_names=[
                'spin_1z',
                'spin_2z'
            ],
            population_parameter_dict={}
        )

    def prob(self, dataset):
        # z component from [-1, 1]
        return 0.25
