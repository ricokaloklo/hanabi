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

    def prob(self, dataset, axis=None):
        if not self._check_if_keys_exist(names=self.signal_parameter_names, keys=dataset.keys()):
            # Parameters needed for evaluation does not exist
            self._parameter_conversion(dataset)

        return self._prob(dataset)

    def _prob(self, dataset, axis=None):
        raise NotImplementedError

    def ln_prob(self, dataset, axis=None):
        return np.log(self.prob(dataset))


class Marginalized(SourcePopulationModel):
    def __init__(self):
        super(Marginalized, self).__init__([], {})

    def _check_if_keys_exist(names, keys):
        return True

    def prob(self, dataset, axis=None):
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

    def _prob(self, dataset, axis=None):
        return gwpopulation.models.mass.power_law_primary_mass_ratio(
            {"mass_1": dataset["mass_1_source"], "mass_ratio": dataset["mass_ratio"]},
            self.population_parameter_dict["alpha"],
            self.population_parameter_dict["beta"],
            self.population_parameter_dict["mmin"],
            self.population_parameter_dict["mmax"]
        )

class PowerLawComponentMass(SourcePopulationModel):
    def __init__(self, alpha, beta, mmin, mmax):
        super(PowerLawComponentMass, self).__init__(
            signal_parameter_names=[
                "mass_1_source",
                "mass_2_source",
            ],
            population_parameter_dict={
                "alpha": alpha,
                "beta": beta,
                "mmin": mmin,
                "mmax": mmax,
            }
        )
        
    def _prob(self, dataset, axis=None):
        return np.exp(self._ln_prob(dataset, axis=axis))
    
    def _ln_prob(self, dataset, axis=None):
        alpha = self.population_parameter_dict["alpha"]
        beta = self.population_parameter_dict["beta"]
        mmin = self.population_parameter_dict["mmin"]
        mmax = self.population_parameter_dict["mmax"]
        m1 = dataset["mass_1_source"]
        m2 = dataset["mass_2_source"]
        
        m1_norm = (1-alpha)/(mmax**(1-alpha) - mmin**(1-alpha))
        m2_norm = (1+beta)/(m1**(1+beta) - mmin**(1+beta))
    
        log_pm1 = -alpha*np.log(m1) + np.log(m1_norm)
        log_pm2 = beta*np.log(m2) + np.log(m2_norm)
        
        return np.where((m2 < m1) & (mmin < m2) & (m1 < mmax), log_pm1 + log_pm2, np.NINF)


class UniformAlignedSpinComponent(SourcePopulationModel):
    def __init__(self):
        super(UniformAlignedSpinComponent, self).__init__(
            signal_parameter_names=[
                'spin_1z',
                'spin_2z'
            ],
            population_parameter_dict={}
        )

    def prob(self, dataset, axis=None):
        # z component from [-1, 1]
        return 0.25

class UniformMagnitudeIsotropicSpin(SourcePopulationModel):
    def __init__(self, a_max=1.0):
        super(UniformMagnitudeIsotropicSpin, self).__init__(
            signal_parameter_names=[
                'a_1',
                'a_2',
                'tilt_1',
                'tilt_2',
                'phi_12',
                'phi_jl'
            ],
            population_parameter_dict={
                "a_max": a_max,
            }
        )

    def _prob(self, dataset, axis=None):
        prior_dict = {
            'a_1': bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=self.population_parameter_dict["a_max"]),
            'a_2': bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=self.population_parameter_dict["a_max"]),
            'tilt_1': bilby.core.prior.Sine(name='tilt_1'),
            'tilt_2': bilby.core.prior.Sine(name='tilt_2'),
            'phi_12': bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'),
            'phi_jl': bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        }

        probs = np.array([prior_dict[p].prob(dataset[p]) for p in self.signal_parameter_names])
    
        return np.prod(probs, axis=0)
