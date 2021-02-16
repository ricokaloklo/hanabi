import numpy as np
from scipy.special import logsumexp
from astropy.cosmology import Planck15
import bilby
from bilby.core.prior import Prior, PriorDict
from .utils import get_ln_weights_for_reweighting
from .marginalized_likelihood import DetectorFrameComponentMassesFromSourceFrame

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
    def __init__(self, result, mass_src_pop_model, spin_src_pop_model, z_src_prob_dist):
        self.result = result

        # Check if redshift is calculated and stored in result.posterior
        if not "redshift" in self.result.posterior.columns:
            self.result.posterior = bilby.gw.conversion.generate_source_frame_parameters(self.result.posterior)

        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        self.z_src_prob_dist = z_src_prob_dist
        self.ln_weights = self.compute_ln_weights()

    def compute_ln_weights_for_component_masses(self, z_src):
        det_frame_priors = DetectorFrameComponentMassesFromSourceFrame(
            self.mass_src_pop_model,
            z_src=z_src
        )

        old_priors = PriorDict(dictionary={
            k: self.result.priors[k] for k in ["mass_1", "mass_2"]
        })

        return get_ln_weights_for_reweighting(self.result, old_priors, det_frame_priors, ["mass_1", "mass_2"])

    def compute_ln_weights_for_luminosity_distances(self):
        old_priors = PriorDict(dictionary={"luminosity_distance": self.result.priors["luminosity_distance"]})
        new_priors = PriorDict(dictionary={"luminosity_distance": LuminosityDistancePriorFromRedshift(self.z_src_prob_dist)})

        return get_ln_weights_for_reweighting(self.result, old_priors, new_priors, ["luminosity_distance"])

    def compute_ln_weights_for_luminosity_distances_from_redshift(self, z_src):
        old_ln_prior_array = self.result.priors["luminosity_distance"].ln_prob(self.result.posterior["luminosity_distance"])
        new_ln_prior_array = np.log(LuminosityDistancePriorFromRedshift(self.z_src_prob_dist).prob_from_z_src(z_src))

        return new_ln_prior_array - old_ln_prior_array

    def compute_ln_weights(self):
        z_src = self.result.posterior["redshift"].astype(np.float64)
        ln_weights = self.compute_ln_weights_for_component_masses(z_src) + \
            self.compute_ln_weights_for_luminosity_distances_from_redshift(z_src)
        
        return ln_weights

    def reweight_ln_evidence(self):
        ln_Z = self.result.log_evidence + logsumexp(self.ln_weights) - np.log(len(self.result.posterior))
        return ln_Z

    def reweight_samples(self):
        reweighted_samples = bilby.result.rejection_sample(self.result.posterior, np.exp(self.ln_weights))
        return reweighted_samples
