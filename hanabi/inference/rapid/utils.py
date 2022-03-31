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
    log_mean_conditional_evidence = logsumexp(log_conditional_evidence) - np.log(len(log_conditional_evidence))
    return base_log_evidence + log_mean_conditional_evidence

def estimate_MonteCarlo_uncertainty(log_fxs):
    N = len(log_fxs)
    # mean = 1/N \sum f(x)
    log_mean = logsumexp(log_fxs) - np.log(N)
    # log squared sum = \sum f(x)^2
    log_squared_sum = logsumexp(2*log_fxs)

    # Monte Carlo Error
    # Note that this is log \sigma^2
    log_variance = -np.log(N) - np.log(N-1) + logsumexp([log_squared_sum, 2*log_mean], b=[1, -N])
    
    log_upper_bound = logsumexp([log_mean, log_variance/2], b=[1,1])
    log_lower_bound = logsumexp([log_mean, log_variance/2], b=[1,-1])
    # Due to the finite numerical accuracy, the two might not be the same
    log_error_from_log_mean = max(abs(log_upper_bound - log_mean), abs(log_mean - log_lower_bound))

    return log_error_from_log_mean

def estimate_uncertainty(base_log_evidence_err, log_conditional_evidence):
    log_mean_conditional_evidence_err = estimate_MonteCarlo_uncertainty(log_conditional_evidence)

    # A crude estimate of the joint uncertainty
    # Note that this implicitly assumes that the error of the base log evidence
    # is independent of the mean conditional evidence
    return np.sqrt(base_log_evidence_err**2 + log_mean_conditional_evidence_err**2)

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

def simulate_run_with_image_type_sampled(result_with_no_img_type, likelihood, ncores=1, save_to_file=True, resample=False):
    simulated_posteriors = {}
    
    # For type-I image, it is the same as the unlensed waveform
    simulated_posteriors["type_I"] = result_with_no_img_type.posterior.copy()
    simulated_posteriors["type_I"]["image_type"] = 1.0 * np.ones(len(simulated_posteriors["type_I"]))
    
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
    simulated_posteriors["type_II"]["image_type"] = 2.0 * np.ones(len(simulated_posteriors["type_II"]))
    
    # For type-III image, it is the same as adding pi/2 to psi
    simulated_posteriors["type_III"] = result_with_no_img_type.posterior.copy()
    simulated_posteriors["type_III"]["psi"] = np.mod(simulated_posteriors["type_III"]["psi"] + np.pi/2., np.pi)
    simulated_posteriors["type_III"]["image_type"] = 3.0 * np.ones(len(simulated_posteriors["type_III"]))

    result = copy.deepcopy(result_with_no_img_type)
    # Concatenating will preserve the posterior pdf at the expensive of storing more samples
    result.posterior = pd.concat(list(simulated_posteriors.values()))
    result.search_parameter_keys.append("image_type")
    if resample:
        result.posterior = result.posterior.sample(len(result_with_no_img_type.posterior))
    result.priors["image_type"] = DiscreteUniform(name="image_type", minimum=1, N=3)

    if save_to_file:
        result.save_to_file(outdir=".", filename="{}_image_type_simulated_result.json".format(result.label))

    return result

setup_logger("hanabi_rapid_analysis")