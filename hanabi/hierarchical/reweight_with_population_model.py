import numpy as np
import logging
from scipy.special import logsumexp
from astropy.cosmology import Planck15
import bilby
from bilby.core.prior import Prior
from .utils import get_ln_weights_for_reweighting, downsample, setup_logger
from .cupy_utils import _GPU_ENABLED, PriorDict
from .marginalized_likelihood import DetectorFrameComponentMassesFromSourceFrame
from ..inference.utils import reweight_log_evidence, estimate_reweighted_log_evidence_err

class LuminosityDistancePriorFromRedshift(Prior):
    def __init__(self, z_src_prob_dist, cosmo=Planck15, name=None, latex_label=None, unit='Mpc'):
        super(LuminosityDistancePriorFromRedshift, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
        )
        self.z_src_prob_dist = z_src_prob_dist
        self.cosmology = cosmo

    def Jacobian(self, z_src):
        d_C = self.cosmology.comoving_distance(z_src).to(self.unit).value
        d_H = self.cosmology.hubble_distance.to(self.unit).value
        E = self.cosmology.efunc(z_src)

        ddL_dz = d_C + (1.+z_src)*d_H/E
        return 1./ddL_dz

    def prob(self, d_L):
        z_src = bilby.gw.conversion.luminosity_distance_to_redshift(d_L) # This is taking a long time
        return self.z_src_prob_dist.prob(z_src)*self.Jacobian(z_src)

    def prob_from_z_src(self, z_src):
        return self.z_src_prob_dist.prob(z_src)*self.Jacobian(z_src)

class ReweightWithPopulationModel(object):
    def __init__(self, result, mass_src_pop_model, spin_src_pop_model, z_src_prob_dist, n_samples=None):
        self.result = result

        # Check if redshift is calculated and stored in result.posterior
        if not "redshift" in self.result.posterior.columns:
            self.result.posterior = bilby.gw.conversion.generate_source_frame_parameters(self.result.posterior)

        # Downsample if n_samples is given
        self.keep_idxs = downsample(len(self.result.posterior), n_samples)
        # Extract only the relevant parameters
        parameters_to_extract = ["luminosity_distance", "mass_1", "mass_2"]
        self.data = {p: self.result.posterior[p].to_numpy()[self.keep_idxs] for p in parameters_to_extract}

        # Evaluate the pdf of the sampling prior once and only once using numpy
        sampling_priors = PriorDict(dictionary={p: self.result.priors[p] for p in parameters_to_extract})
        self.sampling_prior_ln_prob = sampling_priors.ln_prob(self.data, axis=0)
        self.data["redshift"] = self.result.posterior["redshift"].to_numpy(dtype=np.float64)[self.keep_idxs]
         
        logger = logging.getLogger("hanabi_hierarchical_analysis")
        if _GPU_ENABLED:
            # NOTE gwpopulation will automatically use GPU for computation (no way to disable that)
            self.use_gpu = True
            logger.info("Using GPU for reweighting")
            import cupy as cp
            # Move data to GPU
            self.sampling_prior_ln_prob = cp.asarray(self.sampling_prior_ln_prob)
            for k in self.data.keys():
                self.data[k] = cp.asarray(self.data[k])
        else:
            # Fall back to numpy
            self.use_gpu = False
            logger.info("Using CPU for reweighting")

        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        self.z_src_prob_dist = z_src_prob_dist
        self.ln_weights = self.compute_ln_weights()

    def compute_ln_prob_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )
        
        new_prior_pdf = det_frame_priors.ln_prob({k: self.data[k] for k in ["mass_1", "mass_2"]}, axis=0)
        return new_prior_pdf

    def compute_ln_prob_for_luminosity_distances_from_redshift(self, z_src):
        if self.use_gpu:
            import cupy as cp
            # Use CPU instead
            z_src = cp.asnumpy(z_src)

        new_prior_pdf = np.log(LuminosityDistancePriorFromRedshift(self.z_src_prob_dist).prob_from_z_src(z_src))

        if self.use_gpu:
            new_prior_pdf = cp.asarray(new_prior_pdf)

        return new_prior_pdf

    def compute_ln_weights(self):
        ln_weights = self.compute_ln_prob_for_component_masses(self.data["redshift"]) + \
            self.compute_ln_prob_for_luminosity_distances_from_redshift(self.data["redshift"]) - \
            self.sampling_prior_ln_prob
        
        if self.use_gpu:
            import cupy as cp
            ln_weights = cp.asnumpy(ln_weights)
        return ln_weights

    def reweight_ln_evidence(self, estimate_uncertainty=False):
        reweighted_ln_evidence = reweight_log_evidence(self.result.log_evidence, self.ln_weights)

        if estimate_uncertainty:
            reweighted_ln_evidence_err = estimate_reweighted_log_evidence_err(self.result.log_evidence, self.result.log_evidence_err, self.ln_weights)
            return reweighted_ln_evidence, reweighted_ln_evidence_err
        else:
            return reweighted_ln_evidence

    def reweight_samples(self):
        reweighted_samples = bilby.result.rejection_sample(self.result.posterior.iloc[self.keep_idxs], np.exp(self.ln_weights))
        return reweighted_samples
