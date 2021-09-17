import numpy as np
import bilby
import logging
from .utils import get_ln_weights_for_reweighting, downsample, setup_logger
from ..inference.utils import ParameterSuffix
from .cupy_utils import _GPU_ENABLED, PriorDict, logsumexp
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Prior

if _GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

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
class LuminosityDistanceJointPriorFromMagnificationJointDist(object):
    def __init__(self, magnification_joint_distribution, z_src, sep_char="^", suffix=None):
        self.magnification_joint_distribution = magnification_joint_distribution
        self.z_src = z_src
        # Do redshift -> d_L conversion only once
        self.d_L_src = bilby.gw.conversion.redshift_to_luminosity_distance(z_src)
        self.sep_char = sep_char
        if suffix is None:
            self.suffix = ParameterSuffix(self.sep_char)
        else:
            self.suffix = suffix

    def Jacobian(self, mu):
        return 2.0 * mu**1.5 / self.d_L_src

    def prob(self, dataset, axis=None):
        # Convert apparent luminosity distances to absolute magnification first
        mu_abs = {k: (self.d_L_src/dataset["luminosity_distance"+self.suffix(i)])**2 for i, k in enumerate(self.magnification_joint_distribution.keys())}
        # Evaluate joint prior probability (can be independent or conditional)
        joint_prob = self.magnification_joint_distribution.prob(mu_abs, axis=axis)

        overall_Jacobian = 1
        for mu in mu_abs.values():
            overall_Jacobian *= self.Jacobian(mu)

        return joint_prob*overall_Jacobian

    def ln_prob(self, dataset, axis=None):
        return xp.log(self.prob(dataset, axis=axis))

# NOTE This is not a full-blown bilby PriorDict but it does the job!
class DetectorFrameComponentMassesFromSourceFrame(object):
    def __init__(self, mass_src_pop_model, z_src):
        self.mass_src_pop_model = mass_src_pop_model
        self.z_src = z_src

    def Jacobian(self):
        return (1. + self.z_src)**-2

    def prob(self, dataset, axis=None):
        return self.mass_src_pop_model.prob({k+"_source": dataset[k]/(1.+self.z_src) for k in ["mass_1", "mass_2"]}, axis=axis)*self.Jacobian()

    def ln_prob(self, dataset, axis=None):
        return xp.log(self.prob(dataset, axis=axis))

class MonteCarloMarginalizedLikelihood(Likelihood):
    def __init__(self, result, mass_src_pop_model, spin_src_pop_model, magnification_joint_distribution, sampling_priors=None, sep_char="^", suffix=None, n_samples=None):
        # The likelihood is a function of the source redshift only
        # Might as well do this marginalization deterministically
        self.parameters = {'redshift': 0.0}
        self._meta_data = None
        self._marginalized_parameters = []
        self.result = result
        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model

        self.sep_char = sep_char
        if suffix is None:
            self.suffix = ParameterSuffix(self.sep_char)
        else:
            self.suffix = suffix

        # Backward compatibility: previously we expect the 4th argument to be a list of absolute magnification distributions
        if type(magnification_joint_distribution) is list:
            magnification_joint_distribution = {"absolute_magnification" + "_{}".format(i+1): magnification_joint_distribution[i] for i in range(len(magnification_joint_distribution))}
        # NOTE The default suffix ^(_) is not compatible with variable naming rules in python
        for k in list(magnification_joint_distribution.keys()):
            # Check if the default suffix is in the keys
            if not k.isidentifier():
                raise NameError("{} is not a valid variable name in python. Please rename the parameter (e.g. absolute_magnification_1 instead)".format(k))
        self.magnification_joint_distribution = PriorDict(magnification_joint_distribution)

        # Downsample if n_samples is given
        self.keep_idxs = downsample(len(self.result.posterior), n_samples)

        # Extract only the relevant parameters
        parameters_to_extract = ["luminosity_distance" + self.suffix(trigger_idx) for trigger_idx, _ in enumerate(self.magnification_joint_distribution.keys())]
        parameters_to_extract += ["mass_1", "mass_2"]
        self.data = {p: self.result.posterior[p].to_numpy()[self.keep_idxs] for p in parameters_to_extract}

        # Evaluate the pdf of the sampling prior once and only once using numpy
        if sampling_priors is None:
            sampling_priors = bilby.core.prior.PriorDict(dictionary={p: self.result.priors[p] for p in parameters_to_extract})
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
        parameters = ["luminosity_distance" + self.suffix(trigger_idx) for trigger_idx, _ in enumerate(self.magnification_joint_distribution.keys())]
        new_priors = LuminosityDistanceJointPriorFromMagnificationJointDist(
            self.magnification_joint_distribution,
            z_src,
            sep_char=self.sep_char,
            suffix=self.suffix
        )

        return new_priors.ln_prob({p: self.data[p] for p in parameters}, axis=0)

    def compute_ln_prob_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )

        return det_frame_priors.ln_prob({p: self.data[p] for p in ["mass_1", "mass_2"]}, axis=0)

    def log_likelihood(self):
        z_src = float(self.parameters["redshift"])
        ln_weights = self.compute_ln_prob_for_component_masses(z_src) + \
            self.compute_ln_prob_for_luminosity_distances(z_src) - \
            self.sampling_prior_ln_prob
        ln_Z = self.result.log_evidence + logsumexp(ln_weights) - np.log(len(ln_weights))
        return np.nan_to_num(ln_Z, nan=-np.inf)
