import datetime
import json
import os
import pickle
import time
import logging
import copy

import numpy as np
import pandas as pd
from pandas import DataFrame

import bilby
from bilby.gw import conversion
import dynesty
from nestcheck import data_processing
from bilby_pipe.utils import convert_string_to_list

from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool
from parallel_bilby.utils import get_cli_args, stdout_sampling_log
from parallel_bilby.parser import parse_analysis_args
from parallel_bilby.analysis.plotting import plot_current_state
from parallel_bilby.analysis.read_write import (
    read_saved_state,
    write_current_state,
    write_sample_dump,
)
from parallel_bilby.analysis.analysis_run import AnalysisRun
from parallel_bilby.analysis.likelihood import setup_likelihood, reorder_loglikelihoods

from .joint_analysis import JointDataAnalysisInput as JointDataAnalysisInputForBilby
from .parser import create_joint_analysis_pbilby_parser
from .utils import get_version_information
__version__ = get_version_information()
__prog__ = "hanabi_joint_analysis_pbilby"
logger = logging.getLogger(__prog__)

class JointDataAnalysisInput(JointDataAnalysisInputForBilby):
    def initialize_single_trigger_data_analysis_inputs(self):
        self.single_trigger_likelihoods = []
        self.single_trigger_priors = []
        self.single_trigger_args = []
        self.single_trigger_data_dumps = []

        # Loop over data dump files to construct the priors and likelihoods from single triggers
        for idx, data_dump_file in enumerate(self.data_dump_files):
            with open(data_dump_file, "rb") as f:
                data_dump = pickle.load(f)
                self.single_trigger_data_dumps.append(copy.deepcopy(data_dump))
                ifo_list = data_dump["ifo_list"]
                waveform_generator = data_dump["waveform_generator"]
                waveform_generator.start_time = ifo_list[0].time_array[0]
                args = data_dump["args"]
                self.single_trigger_args.append(args)

                priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])
                likelihood = setup_likelihood(
                    interferometers=ifo_list,
                    waveform_generator=waveform_generator,
                    priors=priors,
                    args=args,
                )
                likelihood # Making sure that all the pre-computations are done!
                self.single_trigger_likelihoods.append(likelihood)
                self.single_trigger_priors.append(priors)

        self.single_trigger_data_analysis_inputs = self.single_trigger_args
        self._check_consistency_between_data_analysis_inputs(self.single_trigger_data_analysis_inputs, ["reference_frequency"])

class JointAnalysisRun(AnalysisRun):
    def __init__(
        self,
        joint_data_analysis_input,
        outdir=None,
        label=None,
        dynesty_sample="acceptance-walk",
        nlive=5,
        dynesty_bound="live",
        walks=100,
        maxmcmc=5000,
        naccept=60,
        nact=2,
        facc=0.5,
        min_eff=10,
        enlarge=1.5,
        sampling_seed=0,
        proposals=None,
        bilby_zero_likelihood_mode=False,
    ):
        self.maxmcmc = maxmcmc
        self.nact = nact
        self.naccept = naccept
        self.proposals = convert_string_to_list(proposals)

        # Constructing joint likelihood and joint prior
        likelihood, priors = joint_data_analysis_input.get_likelihood_and_priors()

        # Manipulating the joint priors
        priors.convert_floats_to_delta_functions()
        sampling_keys = []
        for p in priors:
            if isinstance(priors[p], bilby.core.prior.Constraint):
                continue
            elif priors[p].is_fixed:
                likelihood.parameters[p] = priors[p].peak
            else:
                sampling_keys.append(p)

        periodic = []
        reflective = []
        for ii, key in enumerate(sampling_keys):
            if priors[key].boundary == "periodic":
                logger.debug(f"Setting periodic boundary for {key}")
                periodic.append(ii)
            elif priors[key].boundary == "reflective":
                logger.debug(f"Setting reflective boundary for {key}")
                reflective.append(ii)

        if len(periodic) == 0:
            periodic = None
        if len(reflective) == 0:
            reflective = None

        # Setting up the sampler
        self.init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            enlarge=enlarge,
        )
        self._set_sampling_method()

        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for reproducibility
        self.sampling_seed = sampling_seed
        self.rstate = np.random.Generator(np.random.PCG64(self.sampling_seed))
        logger.debug(
            f"Setting random state = {self.rstate} (seed={self.sampling_seed})"
        )

        self.outdir = outdir
        self.label = label
        self.priors = priors
        self.sampling_keys = sampling_keys
        self.likelihood = likelihood
        self.zero_likelihood_mode = bilby_zero_likelihood_mode
        self.periodic = periodic
        self.reflective = reflective
        self.nlive = nlive
        self.joint_data_analysis_input = joint_data_analysis_input

def fill_sample(args):
    """Fill the sample for a particular row in the posterior data frame.

    This function is used inside a pool.map(), so its interface needs
    to be a single argument that is then manually unpacked.

    Parameters
    ----------
    args: tuple
        (row number, row, likelihood)

    Returns
    -------
    sample: array-like

    """
    ii, sample, likelihood = args
    sample = dict(sample).copy()
    # NOTE: We do not do any conversion and re-generation here
    return sample

def format_result(
    run,
    out,
    weights,
    nested_samples,
    sampler_kwargs,
    sampling_time,
    rejection_sample_posterior=True,
):
    """
    Packs the variables from the run into a bilby result object

    Parameters
    ----------
    run: AnalysisRun
        Parallel Bilby run object
    data_dump: str
        Path to the *_data_dump.pickle file
    out: dynesty.results.Results
        Results from the dynesty sampler
    weights: numpy.ndarray
        Array of weights for the points
    nested_samples: pandas.core.frame.DataFrame
        DataFrame of the weights and likelihoods
    sampler_kwargs: dict
        Dictionary of keyword arguments for the sampler
    sampling_time: float
        Time in seconds spent sampling
    rejection_sample_posterior: bool
        Whether to generate the posterior samples by rejection sampling the
        nested samples or resampling with replacement

    Returns
    -------
    result: bilby.core.result.Result
        result object with values written into its attributes
    """

    result = bilby.core.result.Result(
        label=run.label, outdir=run.outdir, search_parameter_keys=run.sampling_keys
    )
    result.priors = run.priors
    result.nested_samples = nested_samples
    result.meta_data["command_line_args"] = vars(run.joint_data_analysis_input.args)
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
    result.meta_data["data_dump"] = run.joint_data_analysis_input.data_dump_files
    result.meta_data["likelihood"] = run.likelihood.meta_data
    result.meta_data["sampler_kwargs"] = run.init_sampler_kwargs
    result.meta_data["run_sampler_kwargs"] = sampler_kwargs
    result.meta_data["injection_parameters"] = {
        data_dump["args"].label: data_dump.get("injection_parameters", None) for data_dump in run.joint_data_analysis_input.single_trigger_data_dumps
    }
    result.injection_parameters = None # The existing infrastructure cannot handle this

    if rejection_sample_posterior:
        keep = weights > np.random.uniform(0, max(weights), len(weights))
        result.samples = out.samples[keep]
        result.log_likelihood_evaluations = out.logl[keep]
        logger.info(
            f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples"
        )
    else:
        result.samples = dynesty.utils.resample_equal(out.samples, weights)
        result.log_likelihood_evaluations = reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl,
            unsorted_samples=out.samples,
            sorted_samples=result.samples,
        )
        logger.info("Resampling nested samples to posterior samples in place.")

    result.log_evidence = out.logz[-1] + run.likelihood.noise_log_likelihood()
    result.log_evidence_err = out.logzerr[-1]
    result.log_noise_evidence = run.likelihood.noise_log_likelihood()
    result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
    result.sampling_time = sampling_time
    result.num_likelihood_evaluations = np.sum(out.ncall)

    result.samples_to_posterior(likelihood=run.likelihood, priors=result.priors)
    return result


def joint_analysis_runner(
    joint_data_analysis_input,
    outdir=None,
    label=None,
    dynesty_sample="acceptance-walk",
    nlive=5,
    dynesty_bound="live",
    walks=100,
    proposals=None,
    maxmcmc=5000,
    naccept=60,
    nact=2,
    facc=0.5,
    min_eff=10,
    enlarge=1.5,
    sampling_seed=0,
    bilby_zero_likelihood_mode=False,
    rejection_sample_posterior=True,
    fast_mpi=False,
    mpi_timing=False,
    mpi_timing_interval=0,
    check_point_deltaT=3600,
    n_effective=np.inf,
    dlogz=10,
    do_not_save_bounds_in_resume=True,
    n_check_point=1000,
    max_its=1e10,
    max_run_time=1e10,
    rotate_checkpoints=False,
    no_plot=False,
    nestcheck=False,
    result_format="hdf5",
    **kwargs,
):
    # Initialise a run
    run = JointAnalysisRun(
        joint_data_analysis_input,
        outdir=outdir,
        label=label,
        dynesty_sample=dynesty_sample,
        nlive=nlive,
        dynesty_bound=dynesty_bound,
        walks=walks,
        maxmcmc=maxmcmc,
        nact=nact,
        naccept=naccept,
        facc=facc,
        min_eff=min_eff,
        enlarge=enlarge,
        sampling_seed=sampling_seed,
        proposals=proposals,
        bilby_zero_likelihood_mode=bilby_zero_likelihood_mode,
    )

    t0 = datetime.datetime.now()
    sampling_time = 0
    with MPIPool(
        parallel_comms=fast_mpi,
        time_mpi=mpi_timing,
        timing_interval=mpi_timing_interval,
        use_dill=True,
    ) as pool:
        if pool.is_master():
            POOL_SIZE = pool.size

            logger.info(f"sampling_keys={run.sampling_keys}")
            if run.periodic:
                logger.info(
                    f"Periodic keys: {[run.sampling_keys[ii] for ii in run.periodic]}"
                )
            if run.reflective:
                logger.info(
                    f"Reflective keys: {[run.sampling_keys[ii] for ii in run.reflective]}"
                )
            logger.info("Using priors:")
            for key in run.priors:
                logger.info(f"{key}: {run.priors[key]}")

            resume_file = f"{run.outdir}/{run.label}_checkpoint_resume.pickle"
            samples_file = f"{run.outdir}/{run.label}_samples.dat"

            sampler, sampling_time = read_saved_state(resume_file)

            if sampler is False:
                logger.info(f"Initializing sampling points with pool size={POOL_SIZE}")
                live_points = run.get_initial_points_from_prior(pool)
                logger.info(
                    f"Initialize NestedSampler with "
                    f"{json.dumps(run.init_sampler_kwargs, indent=1, sort_keys=True)}"
                )
                sampler = run.get_nested_sampler(live_points, pool, POOL_SIZE)
            else:
                # Reinstate the pool and map (not saved in the pickle)
                logger.info(f"Read in resume file with sampling_time = {sampling_time}")
                sampler.pool = pool
                sampler.M = pool.map
                sampler.loglikelihood.pool = pool

            logger.info(
                f"Starting sampling for job {run.label}, with pool size={POOL_SIZE} "
                f"and check_point_deltaT={check_point_deltaT}"
            )

            sampler_kwargs = dict(
                n_effective=n_effective,
                dlogz=dlogz,
                save_bounds=not do_not_save_bounds_in_resume,
            )
            logger.info(f"Run criteria: {json.dumps(sampler_kwargs)}")

            run_time = 0
            early_stop = False

            for it, res in enumerate(sampler.sample(**sampler_kwargs)):
                stdout_sampling_log(
                    results=res, niter=it, ncall=sampler.ncall, dlogz=dlogz
                )

                iteration_time = (datetime.datetime.now() - t0).total_seconds()
                t0 = datetime.datetime.now()

                sampling_time += iteration_time
                run_time += iteration_time

                if os.path.isfile(resume_file):
                    last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
                else:
                    last_checkpoint_s = np.inf

                """
                Criteria for writing checkpoints:
                a) time since last checkpoint > check_point_deltaT
                b) reached an integer multiple of n_check_point
                c) reached max iterations
                d) reached max runtime
                """

                if (
                    last_checkpoint_s > check_point_deltaT
                    or (it % n_check_point == 0 and it != 0)
                    or it == max_its
                    or run_time > max_run_time
                ):

                    write_current_state(
                        sampler,
                        resume_file,
                        sampling_time,
                        rotate_checkpoints,
                    )
                    write_sample_dump(sampler, samples_file, run.sampling_keys)
                    if no_plot is False:
                        plot_current_state(
                            sampler, run.sampling_keys, run.outdir, run.label
                        )

                    if it == max_its:
                        exit_reason = 1
                        logger.info(
                            f"Max iterations ({it}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break

                    if run_time > max_run_time:
                        exit_reason = 2
                        logger.info(
                            f"Max run time ({max_run_time}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break

            if not early_stop:
                exit_reason = 0
                # Adding the final set of live points.
                for it_final, res in enumerate(sampler.add_live_points()):
                    pass

                # Create a final checkpoint and set of plots
                write_current_state(
                    sampler, resume_file, sampling_time, rotate_checkpoints
                )
                write_sample_dump(sampler, samples_file, run.sampling_keys)
                if no_plot is False:
                    plot_current_state(
                        sampler, run.sampling_keys, run.outdir, run.label
                    )

                sampling_time += (datetime.datetime.now() - t0).total_seconds()

                out = sampler.results

                if nestcheck is True:
                    logger.info("Creating nestcheck files")
                    ns_run = data_processing.process_dynesty_run(out)
                    nestcheck_path = os.path.join(run.outdir, "Nestcheck")
                    bilby.core.utils.check_directory_exists_and_if_not_mkdir(
                        nestcheck_path
                    )
                    nestcheck_result = f"{nestcheck_path}/{run.label}_nestcheck.pickle"

                    with open(nestcheck_result, "wb") as file_nest:
                        pickle.dump(ns_run, file_nest)

                weights = np.exp(out["logwt"] - out["logz"][-1])
                nested_samples = DataFrame(out.samples, columns=run.sampling_keys)
                nested_samples["weights"] = weights
                nested_samples["log_likelihood"] = out.logl

                result = format_result(
                    run,
                    out,
                    weights,
                    nested_samples,
                    sampler_kwargs,
                    sampling_time,
                )

                posterior = conversion.fill_from_fixed_priors(
                    result.posterior, run.priors
                )

                logger.info(
                    "Generating posterior from marginalized parameters for"
                    f" nsamples={len(posterior)}, POOL={pool.size}"
                )
                fill_args = [
                    (ii, row, run.likelihood) for ii, row in posterior.iterrows()
                ]
                samples = pool.map(fill_sample, fill_args)
                result.posterior = pd.DataFrame(samples)

                logger.debug(
                    "Updating prior to the actual prior (undoing marginalization)"
                )
                for par, name in zip(
                    ["distance", "phase", "time"],
                    ["luminosity_distance", "phase", "geocent_time"],
                ):
                    if getattr(run.likelihood, f"{par}_marginalization", False):
                        run.priors[name] = run.likelihood.priors[name]
                result.priors = run.priors

                result.posterior = result.posterior.applymap(
                    lambda x: x[0] if isinstance(x, list) else x
                )
                result.posterior = result.posterior.select_dtypes([np.number])
                logger.info(
                    f"Saving result to {run.outdir}/{run.label}_result.{result_format}"
                )
                if result_format != "json":  # json is saved by default
                    result.save_to_file(extension="json")
                result.save_to_file(extension=result_format)
                print(
                    f"Sampling time = {datetime.timedelta(seconds=result.sampling_time)}s"
                )
                print(f"Number of lnl calls = {result.num_likelihood_evaluations}")
                print(result)
                # Disable corner plot generating

        else:
            exit_reason = -1
        return exit_reason

def main():
    cli_args = get_cli_args()

    analysis_parser = create_joint_analysis_pbilby_parser(__prog__, __version__)
    input_args = parse_analysis_args(analysis_parser, cli_args=cli_args)

    joint_data_analysis_input = JointDataAnalysisInput(input_args, [])
    joint_analysis_runner(joint_data_analysis_input=joint_data_analysis_input, **vars(input_args))