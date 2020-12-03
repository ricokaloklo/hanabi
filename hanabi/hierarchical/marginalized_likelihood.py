import numpy as np
import bilby
from .utils import get_ln_weights_for_reweighting
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Prior, PriorDict
from scipy.special import logsumexp

class LuminosityDistancePriorFromAbsoluteMagnificationRedshift(Prior):
    def __init__(self, abs_magnification_prob_dist, z_src, name=None, latex_label=None, unit=None):
        super(LuminosityDistancePriorFromAbsoluteMagnificationRedshift, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
        )
        self.abs_magnification_prob_dist = abs_magnification_prob_dist
        self.z_src = z_src
        self.d_L_src = bilby.gw.conversion.redshift_to_luminosity_distance(z_src)

    def Jacobian(self, d_L):
        return 2.0 * (self.d_L_src/d_L)**2 * (1./d_L)

    def prob(self, d_L):
        mu_abs = (self.d_L_src/d_L)**2
        return self.abs_magnification_prob_dist.prob(mu_abs)*self.Jacobian(d_L)

# NOTE This is not a full-blown bilby PriorDict but it does the job!
class DetectorFrameComponentMassesFromSourceFrame(dict):
    def __init__(self, mass_src_pop_model, z_src):
        self.mass_src_pop_model = mass_src_pop_model
        self.z_src = z_src

    def Jacobian(self):
        return np.power((1. + self.z_src), -2)

    def prob(self, dataset):
        return self.mass_src_pop_model.prob({k+"_source": dataset[k]/(1.+self.z_src) for k in ["mass_1", "mass_2"]})

    def ln_prob(self, dataset):
        return np.log(self.prob(dataset))

class MonteCarloMarginalizedLikelihood(Likelihood):
    def __init__(self, result, mass_src_pop_model, spin_src_pop_model, abs_magnification_prob_dists, sep_char="^", suffix=None):
        # The likelihood is a function of the source redshift only
        # Might as well do this marginalization deterministically
        self.parameters = {'redshift': 0.0}
        self.result = result
        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        # This should be a list of abs_magnification_prob_dist
        self.abs_magnification_prob_dists = abs_magnification_prob_dists

        self.sep_char = sep_char
        if suffix is None:
            self.suffix = lambda trigger_idx: "{}({})".format(self.sep_char, trigger_idx + 1)
        else:
            self.suffix = suffix

    def compute_ln_weights_for_luminosity_distances(self, z_src):
        # Construct the prior dicts for apparent luminosity distance
        new_priors = {}
        old_priors = {}
        parameters = []

        for trigger_idx, abs_magn in enumerate(self.abs_magnification_prob_dists):
            parameter_name = "luminosity_distance"+self.suffix(trigger_idx)
            parameters.append(parameter_name)
            new_priors[parameter_name] = \
                LuminosityDistancePriorFromAbsoluteMagnificationRedshift(
                    abs_magnification_prob_dist=abs_magn,
                    z_src=z_src,
                    name=parameter_name,
                    unit="Mpc",
                )
            old_priors[parameter_name] = \
                self.result.priors[parameter_name]

        new_priors = PriorDict(dictionary=new_priors)
        old_priors = PriorDict(dictionary=old_priors)

        return get_ln_weights_for_reweighting(self.result, old_priors, new_priors, parameters)

    def compute_ln_weights_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )

        old_priors = PriorDict(dictionary={
            k: self.result.priors[k] for k in ["mass_1", "mass_2"]
        })

        return get_ln_weights_for_reweighting(self.result, old_priors, det_frame_priors, ["mass_1", "mass_2"])

    def log_likelihood(self):
        z_src = self.parameters["redshift"]
        ln_weights = self.compute_ln_weights_for_component_masses(z_src) + \
            self.compute_ln_weights_for_luminosity_distances(z_src)
        ln_Z = self.result.log_evidence + logsumexp(ln_weights) - np.log(len(self.result.posterior))

        return ln_Z
