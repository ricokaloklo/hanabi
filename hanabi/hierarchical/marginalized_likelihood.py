import numpy as np
import bilby
import logging
from .utils import get_ln_weights_for_reweighting, downsample, setup_logger
from ..inference.utils import ParameterSuffix
from .cupy_utils import _GPU_ENABLED, PriorDict, logsumexp
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Prior

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
        self.keep_idxs = downsample(len(self.result.posterior), n_samples)

        # Extract only the relevant parameters
        parameters_to_extract = ["luminosity_distance" + self.suffix(trigger_idx) for trigger_idx, _ in enumerate(self.abs_magnification_prob_dists)]
        parameters_to_extract += ["mass_1", "mass_2"]
        self.data = {p: self.result.posterior[p].to_numpy()[self.keep_idxs] for p in parameters_to_extract}

        # Evaluate the pdf of the sampling prior once and only once using numpy
        sampling_priors = PriorDict(dictionary={p: self.result.priors[p] for p in parameters_to_extract})
        self.sampling_prior_ln_prob = sampling_priors.ln_prob(self.data, axis=0)

        logger = logging.getLogger("hanabi_hierarchical_analysis")
        if _GPU_ENABLED:
            # NOTE gwpopulation will automatically use GPU for computation (no way to disable that)
            self.use_gpu = True
            logger.info("Using GPU for likelihood evaluation")
            import cupy as cp
            # Move data to GPU
            self.sampling_prior_ln_prob = cp.asarray(self.sampling_prior_ln_prob)
            for k in self.data.keys():
                self.data[k] = cp.asarray(self.data[k])
        else:
            # Fall back to numpy
            self.use_gpu = False
            logger.info("Using CPU for likelihood evaluation")

    def compute_ln_prob_for_luminosity_distances(self, z_src):
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

        return PriorDict(dictionary=new_priors).ln_prob({p: self.data[p] for p in parameters}, axis=0)

    def compute_ln_prob_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )

        return det_frame_priors.ln_prob({p: self.data[p] for p in ["mass_1", "mass_2"]})

    def log_likelihood(self):
        z_src = self.parameters["redshift"]
        ln_weights = self.compute_ln_prob_for_component_masses(z_src) + \
            self.compute_ln_prob_for_luminosity_distances(z_src) - \
            self.sampling_prior_ln_prob
        ln_Z = self.result.log_evidence + logsumexp(ln_weights) - np.log(len(ln_weights))

        return ln_Z
