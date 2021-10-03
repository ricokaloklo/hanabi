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

def setup_likelihood_from_pbilby(interferometers, waveform_generator, priors, args):
    """Takes the kwargs and sets up and returns  either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        The waveform generation
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        The likelihood (either GravitationalWaveTransient or ROQGravitationalWaveTransient)

    """

    likelihood_kwargs = dict(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=args.phase_marginalization,
        distance_marginalization=args.distance_marginalization,
        distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
        time_marginalization=args.time_marginalization,
        reference_frame=args.reference_frame,
        time_reference=args.time_reference,
    )

    if args.likelihood_type == "GravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.GravitationalWaveTransient
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient

        if args.time_marginalization:
            logger.warning(
                "Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"
            )

        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))
    elif "." in args.likelihood_type:
        split_path = args.likelihood_type.split(".")
        module = ".".join(split_path[:-1])
        likelihood_class = split_path[-1]
        Likelihood = getattr(import_module(module), likelihood_class)
        likelihood_kwargs.update(args.extra_likelihood_kwargs)
        if "roq" in args.likelihood_type.lower():
            likelihood_kwargs.pop("time_marginalization", None)
            likelihood_kwargs.pop("jitter_time", None)
            likelihood_kwargs.update(args.roq_likelihood_kwargs)
    else:
        raise ValueError("Unknown Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood

def load_run_from_pbilby(result_file, data_dump_file, **kwargs):
    result = bilby.result.read_in_result(result_file)
    with open(data_dump_file, "rb") as f:
        data_dump = pickle.load(f)

    ifo_list = data_dump["ifo_list"]
    waveform_generator = data_dump["waveform_generator"]
    waveform_generator.start_time = ifo_list[0].time_array[0]
    args = data_dump["args"]
    priors = result.priors

    # Override args if given in kwargs
    for k, v in kwargs.items():
        setattr(args, k, v)

    likelihood = setup_likelihood_from_pbilby(
        interferometers=ifo_list,
        waveform_generator=waveform_generator,
        priors=priors,
        args=args,
    )

    return likelihood, priors, result

def load_run_from_bilby(result_file, data_dump_file, trigger_ini_file, **kwargs):
    result = bilby.result.read_in_result(result_file)
    args, unknown_args = bilby_pipe.utils.parse_args([trigger_ini_file, "--data-dump-file", data_dump_file], bilby_pipe.data_analysis.create_analysis_parser())

    # Override args if given in kwargs
    for k, v in kwargs.items():
        setattr(args, k, v)

    single_trigger_analysis = SingleTriggerDataAnalysisInput(args, unknown_args)
    likelihood, priors = single_trigger_analysis.get_likelihood_and_priors()

    return likelihood, priors, result

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