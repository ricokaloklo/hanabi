import numpy as np
import pandas as pd
import bilby
import bilby_pipe
import inspect
import pickle
from importlib import import_module
from scipy.special import logsumexp
from scipy.optimize import fsolve
import copy
import tqdm
from schwimmbad import SerialPool, MultiPool

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

def _search_for_degenerate_psi(row, l):
    # A helper function to do this search
    def logL_from_logL0(psi, logL0, l):
        l.parameters.update({"image_type": 1.0, "psi": psi})
        return l.log_likelihood() - logL0
    
    l.parameters.update(row[l.priors.keys()])
    # Compute the likelihood as if it is a type-II image
    l.parameters.update({"image_type": 2.0})
    logL0 = l.log_likelihood()
    # Search for another psi that would give the exact same waveform
    new_psi = np.mod(fsolve(logL_from_logL0, row["psi"], args=(logL0, l))[0], np.pi)
    new_row = row.copy().to_dict()
    new_row["psi"] = new_psi
    return new_row

def simulate_run_with_image_type_sampled(result_with_no_img_type, likelihood, ncores=1, resample=False):
    simulated_posteriors = {}
    
    # For type-I image, it is the same as the unlensed waveform
    simulated_posteriors["type_I"] = result_with_no_img_type.posterior.copy()
    
    # For type-II image, we search for the psi that would produce the exact same log likelihood 
    # (and hence same waveform as seen in detector)
    simulated_posteriors["type_II"] = result_with_no_img_type.posterior.copy()
    
    # Use multi-processing
    with MultiPool(ncores) as pool:
        out = pool.starmap(
            _search_for_degenerate_psi,
            tqdm.tqdm([[
                simulated_posteriors["type_II"].iloc[i],
                likelihood,
            ] for i in range(len(simulated_posteriors["type_II"]))])
        )
    out = pd.DataFrame(out)
    simulated_posteriors["type_II"] = out
    
    # For type-III image, it is the same as adding pi/2 to psi
    simulated_posteriors["type_III"] = result_with_no_img_type.posterior.copy()
    simulated_posteriors["type_III"]["psi"] = np.mod(result_with_no_img_type["psi"] + np.pi/2., np.pi)

    result = copy.deepcopy(result_with_no_img_type)
    # Concatenating will preserve the posterior pdf at the expensive of storing more samples
    result.posterior = pd.concat(list(simulated_posteriors.values()))
    if resample:
        result.posterior = result.posterior.sample(len(new_posteriors[0]))
    result.priors["image_type"] = DiscreteUniform(name="image_type", minimum=1, N=3)

    return result

setup_logger("hanabi_rapid_analysis")