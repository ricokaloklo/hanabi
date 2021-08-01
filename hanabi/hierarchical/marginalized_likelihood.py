import numpy as np
import bilby
from .utils import get_ln_weights_for_reweighting
from ..inference.utils import ParameterSuffix
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
class DetectorFrameComponentMassesFromSourceFrame(object):
    def __init__(self, mass_src_pop_model, z_src):
        self.mass_src_pop_model = mass_src_pop_model
        self.z_src = z_src

    def Jacobian(self):
        return np.power((1. + self.z_src), -2)

    def prob(self, dataset, axis=None):
        return self.mass_src_pop_model.prob({k+"_source": dataset[k]/(1.+self.z_src) for k in ["mass_1", "mass_2"]}, axis=axis)*self.Jacobian()

    def ln_prob(self, dataset, axis=None):
        return np.log(self.prob(dataset, axis=axis))

class MonteCarloMarginalizedLikelihood(Likelihood):
    def __init__(self, result, mass_src_pop_model, spin_src_pop_model, abs_magnification_prob_dists, sep_char="^", suffix=None, n_samples=None):
        # The likelihood is a function of the source redshift only
        # Might as well do this marginalization deterministically
        self.parameters = {'redshift': 0.0}
        self._meta_data = None
        self._marginalized_parameters = []
        self.result = result
        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        # This should be a list of abs_magnification_prob_dist
        self.abs_magnification_prob_dists = abs_magnification_prob_dists

        self.sep_char = sep_char
        if suffix is None:
            self.suffix = ParameterSuffix(self.sep_char)
        else:
            self.suffix = suffix

        # Downsample if n_samples is given
        keep_idxs = np.arange(len(self.result.posterior))
        if n_samples is not None:
            keep_idxs = np.random.choice(keep_idxs, size=n_samples)
        self.keep_idxs = keep_idxs

        # Extract only the relevant parameters
        parameters_to_extract = ["luminosity_distance" + self.suffix(trigger_idx) for trigger_idx, _ in enumerate(self.abs_magnification_prob_dists)]
        parameters_to_extract += ["mass_1", "mass_2"]
        self.data = {p: self.result.posterior[p].to_numpy()[keep_idxs] for p in parameters_to_extract}

        # Evaluate the pdf of the sampling prior once and only once using numpy
        sampling_priors = PriorDict(dictionary={p: self.result.priors[p] for p in parameters_to_extract})
        self.sampling_prior_pdf = sampling_priors.ln_prob(self.data, axis=0)

    def compute_ln_weights_for_luminosity_distances(self, z_src):
        # Construct the prior dict for apparent luminosity distance
        new_priors = {}
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

        new_prior_pdf = PriorDict(dictionary=new_priors).ln_prob({p: self.data[p] for p in parameters}, axis=0)
        return new_prior_pdf - self.sampling_prior_pdf

    def compute_ln_weights_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )

        new_prior_pdf = det_frame_priors.ln_prob({p: self.data[p] for p in ["mass_1", "mass_2"]})
        return new_prior_pdf - self.sampling_prior_pdf

    def log_likelihood(self):
        z_src = self.parameters["redshift"]
        ln_weights = self.compute_ln_weights_for_component_masses(z_src) + \
            self.compute_ln_weights_for_luminosity_distances(z_src)
        ln_Z = self.result.log_evidence + logsumexp(ln_weights) - np.log(len(self.result.posterior))

        return ln_Z
