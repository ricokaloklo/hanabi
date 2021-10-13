import numpy as np
import pandas as pd
import bilby
import bilby_pipe
import inspect
import pickle
from importlib import import_module
from scipy.special import logsumexp
import copy

from bilby.gw.likelihood import GravitationalWaveTransient
from ..joint_analysis import SingleTriggerDataAnalysisInput
from ..utils import setup_logger
from ...lensing.prior import DiscreteUniform

_dist_marg_lookup_table_filename_template = ".distance_marginalization_lookup_trigger_{}.npz"

def compute_log_likelihood_for_theta(likelihood, theta):
    likelihood.parameters.update(theta)
    return likelihood.log_likelihood()

def compute_log_joint_evidence_from_log_conditional_evidence(base_log_evidence, log_conditional_evidence):
    return base_log_evidence + logsumexp(log_conditional_evidence) - np.log(len(log_conditional_evidence))

def bootstrap_uncertainty(log_conditional_evidence, n_frac=0.5, n_resample=1000):   
    bootstrapped_estimate = []
    for i in range(n_resample):
        log_ev = compute_log_joint_evidence_from_log_conditional_evidence(
            0.,
            np.random.choice(log_conditional_evidence, size=int(n_frac*len(log_conditional_evidence)), replace=True)
        )
        bootstrapped_estimate.append(log_ev)

    return np.std(bootstrapped_estimate), np.array(bootstrapped_estimate)

def simulate_run_with_image_type_sampled(result_with_no_img_type, resample=False):
    possible_image_types = [1.0, 2.0, 3.0]
    new_posteriors = []

    for img_type in possible_image_types:
        new_posterior = result_with_no_img_type.posterior.copy()
        new_posterior["image_type"] = img_type * np.ones(len(new_posterior))
        new_posterior["psi"] = np.mod(new_posterior["psi"] + (img_type - 1.)*np.pi/4., np.pi)
        new_posteriors.append(new_posterior)

    result = copy.deepcopy(result_with_no_img_type)
    # Concatenating will preserve the posterior pdf at the expensive of storing more samples
    result.posterior = pd.concat(new_posteriors)
    if resample:
        result.posterior = result.posterior.sample(len(new_posteriors[0]))
    result.priors["image_type"] = DiscreteUniform(name="image_type", minimum=1, N=3)

    return result

setup_logger("hanabi_rapid_analysis")