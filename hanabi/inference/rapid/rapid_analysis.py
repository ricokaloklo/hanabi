import sys
import copy
import itertools
import logging
import tqdm
import numpy as np
import pandas as pd
import bilby
import bilby_pipe
from scipy.special import logsumexp
from schwimmbad import SerialPool, MultiPool
from dynesty.utils import resample_equal

from .sampler import sample_time_dist_marginalized
from .parser import create_rapid_analysis_parser
from .utils import _dist_marg_lookup_table_filename_template
from .utils import compute_log_likelihood_for_theta, compute_log_joint_evidence_from_log_conditional_evidence, bootstrap_uncertainty
from .utils import simulate_run_with_image_type_sampled
from .likelihood import SingleLikelihoodWithTransformableWaveformCache
from ..utils import ParameterSuffix, load_run_from_bilby, load_run_from_pbilby
from ...lensing.likelihood import LensingJointLikelihood, LensingJointLikelihoodWithWaveformCache

from ..utils import get_version_information
__version__ = get_version_information()
__prog__ = "hanabi_rapid_analysis"


class RapidAnalysisInput(bilby_pipe.input.Input):
    def __init__(self, args, unknown_args, test=False):
        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        # Read the rest of the supported arguments
        for name in dir(args):
            if not name.startswith("_"):
                setattr(self, name, getattr(args, name, None))
        self.parse_lensing_prior_dict()

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"
        assert self.n_triggers == len(self.data_dump_files), "n_triggers does not match with the number of data dump files given"
        assert self.n_triggers == len(self.result_files), "n_triggers does not match with the number of result files given"
        assert self.n_triggers == len(self.inference_software), "n_triggers does not match with the number of inference software given"

        logger = logging.getLogger(__prog__)
        if self.time_marginalization:
            logger.info("Using time marginalization")
        if self.distance_marginalization:
            logger.info("Using distance marginalization")

        # FIXME Make sure no one will use the ^ sign to name actual parameter
        self.sep_char = "^"
        self.suffix = ParameterSuffix(self.sep_char)

        self.trigger_ids = np.arange(self.n_triggers)
        if self.symmetrize:
            self.trigger_combo = [np.roll(self.trigger_ids, -s) for s in range(self.n_triggers)]
            self.trigger_combo_weight = np.ones(self.n_triggers, dtype=float)/self.n_triggers
        else:
            self.trigger_combo = [self.trigger_ids]
            self.trigger_combo_weight = np.array([1.])

        # Load and reconstruct likelihood for evaluation
        self.initialize_single_trigger_inference_inputs()

        if self.downsample is None:
            self.n_posterior = min([len(r.posterior) for r in self.single_trigger_results])
        else:
            # FIXME This assumes that the given --downsample is sane
            self.n_posterior = self.downsample
        self.n_cores = self.request_cpus

    def parse_lensing_prior_dict(self):
        # Pre-process/convert the prior dict string
        lensing_prior_dict = bilby_pipe.utils.convert_prior_string_input(self.lensing_prior_dict)
        lensing_prior_dict = {
            bilby_pipe.input.Input._convert_prior_dict_key(key): val for key, val in lensing_prior_dict.items()
        }
        self.lensing_prior_dict = bilby.core.prior.PriorDict(lensing_prior_dict)

    def initialize_single_trigger_inference_inputs(self):
        self.single_trigger_likelihoods = []
        self.single_trigger_likelihoods_with_cache = []
        self.single_trigger_priors = []
        self.single_trigger_results = []

        _disable_all_marginalization = {
            "time_marginalization": False,
            "distance_marginalization": False,
            "phase_marginalization": False,
        }

        for idx, (trigger_ini_file, data_dump_file, result_file, PE_program) in enumerate(zip(self.trigger_ini_files, self.data_dump_files, self.result_files, self.inference_software)):
            if PE_program == "bilby":
                single_trigger_likelihood, priors, single_trigger_result = \
                    load_run_from_bilby(
                        data_dump_file,
                        trigger_ini_file,
                        result_file=result_file,
                        calibration_model=None,
                        **_disable_all_marginalization,
                    )                
            elif PE_program == "parallel_bilby":
                single_trigger_likelihood, priors, single_trigger_result = \
                    load_run_from_pbilby(
                        data_dump_file,
                        result_file=result_file,
                        calibration_model=None,
                        **_disable_all_marginalization,
                    )
            else:
                raise ValueError("Cannot recognize/does not support the inference software {}".format(PE_program))


            # FIXME Compatibility issue
            for ifo in single_trigger_likelihood.interferometers:
                ifo.calibration_model = bilby.gw.detector.calibration.Recalibrate()
                ifo.strain_data.notch_list = []

            single_trigger_likelihood_with_cache = SingleLikelihoodWithTransformableWaveformCache.from_likelihood(
                single_trigger_likelihood,
                time_marginalization=self.time_marginalization,
                distance_marginalization=self.distance_marginalization,
                distance_marginalization_lookup_table=_dist_marg_lookup_table_filename_template.format(idx+1)
            )
            if self.time_marginalization:
                # Necessary to make time marginalization works
                single_trigger_likelihood_with_cache.parameters.update({
                    "geocent_time": float(single_trigger_likelihood_with_cache.interferometers.start_time)
                })

            self.single_trigger_likelihoods.append(single_trigger_likelihood)
            self.single_trigger_likelihoods_with_cache.append(single_trigger_likelihood_with_cache)
            self.single_trigger_priors.append(priors)
            self.single_trigger_results.append(single_trigger_result)

            # Check if image_type is being sampled over
            for idx in range(len(self.single_trigger_results)):
                r = self.single_trigger_results[idx]

                if "image_type" not in r.search_parameter_keys:
                    # Simulate run with image_type sampled by shifting the polarization angle
                    logger = logging.getLogger(__prog__)
                    logger.info("\"image type\" is not being sampled in {}. Simulating an inference with this sampled".format(r.label))
                    self.single_trigger_results[idx] = simulate_run_with_image_type_sampled(r, resample=True)
                    # Update also the priors
                    self.single_trigger_priors[idx] = self.single_trigger_results[idx].priors

    def combine_runs(self, conditional_inference_results):
        # FIXME Before combining we should check and see if the runs are 'compatible' -- having same priors, etc
        if self.generate_posterior_samples:
            combined_posterior_samples = pd.concat([r.posterior for r in conditional_inference_results])
        else:
            combined_posterior_samples = None
        combined_log_evidence = logsumexp([r.log_evidence for r in conditional_inference_results], b=self.trigger_combo_weight)
        joint_priors = conditional_inference_results[0].priors

        combined_result = bilby.core.result.Result(
            outdir=self.outdir,
            label="{label}_combined".format(label=self.label),
            sampler="hanabi_rapid_analysis",
            search_parameter_keys=[k for k in list(joint_priors.keys()) if type(joint_priors[k]) != bilby.core.prior.Constraint],
            posterior=combined_posterior_samples,
            log_evidence=combined_log_evidence,
            priors=joint_priors
        )

        combined_result.save_to_file(outdir=self.outdir)

    def run_analysis(self):
        conditional_inference_results = []

        for combo in self.trigger_combo:
            inference = ConditionalInference(
                combo,
                self.single_trigger_likelihoods,
                self.single_trigger_likelihoods_with_cache,
                self.single_trigger_priors,
                self.single_trigger_results,
                self.n_posterior,
                self.common_parameters,
                self.lensing_prior_dict,
                sep_char=self.sep_char,
                suffix=self.suffix,
                outdir=self.outdir,
                label=self.label,
                n_cores=self.n_cores,
                waveform_cache=self.waveform_cache,
                generate_posterior_samples=self.generate_posterior_samples,
            )

            logger = logging.getLogger(__prog__)
            logger.info("Running inference on {} conditioned on {}".format(
                " ,".join([self.single_trigger_results[i].label for i in combo[1:]]),
                self.single_trigger_results[combo[0]].label
            ))

            conditional_inference_results.append(inference.run())

        if len(self.trigger_combo) > 1:
            self.combine_runs(conditional_inference_results)

class ConditionalInference():
    def __init__(
            self,
            trigger_ids,
            single_trigger_likelihoods,
            single_trigger_likelihoods_with_cache,
            single_trigger_priors,
            single_trigger_results,
            n_posterior,
            common_parameters,
            lensing_prior_dict,
            sep_char="^",
            suffix=None,
            n_cores=1,
            outdir="outdir",
            label="label",
            waveform_cache=True,
            generate_posterior_samples=False,
        ):
        self.trigger_ids = trigger_ids
        self.single_trigger_likelihoods = single_trigger_likelihoods
        self.single_trigger_likelihoods_with_cache = single_trigger_likelihoods_with_cache
        self.single_trigger_priors = single_trigger_priors
        self.single_trigger_results = single_trigger_results
        self.n_posterior = n_posterior
        self.common_parameters = common_parameters
        self.lensing_prior_dict = lensing_prior_dict
        self.n_cores = n_cores
        self.outdir = outdir
        self.label = label
        self.waveform_cache = waveform_cache
        self.generate_posterior_samples = generate_posterior_samples

        self.sep_char = sep_char
        if suffix is None:
            suffix = ParameterSuffix(self.sep_char)
        self.suffix = suffix

        # Modify the label to record which trigger the inference is conditioned on
        self.label = "{label}_conditioned_on_{base_label}".format(label=self.label, base_label=self.single_trigger_results[self.trigger_ids[0]].label)

        self.likelihood_base = self.single_trigger_likelihoods[self.trigger_ids[0]] # Condition on this trigger
        self.likelihood_base_with_cache = self.single_trigger_likelihoods_with_cache[self.trigger_ids[0]]
        # NOTE This assumes that all the likelihood functions take the same input parameters
        self.likelihood_parameter_keys = \
            [p for p in list(self.likelihood_base.priors.sample().keys()) if not p.startswith("recalib")]
        self.independent_parameters = [p for p in self.likelihood_parameter_keys if p not in self.common_parameters]

        # Reconstruct the effective joint prior
        # The priors for the common parameters are from the base trigger
        self.joint_priors = {k: self.single_trigger_priors[self.trigger_ids[0]][k] for k in self.common_parameters}
        for trigger_idx in self.trigger_ids:
            for p in self.independent_parameters:
                if p.startswith("recalib"):
                    # Ignore calibration uncertainty
                    continue
                self.joint_priors[p+self.suffix(trigger_idx)] = copy.deepcopy(self.single_trigger_priors[trigger_idx][p])
                # Rename the parameter
                self.joint_priors[p+self.suffix(trigger_idx)].name += self.suffix(trigger_idx)
                # LaTeX label
                if self.joint_priors[p+self.suffix(trigger_idx)].latex_label.endswith("$"):
                    self.joint_priors[p+self.suffix(trigger_idx)].latex_label = \
                        self.joint_priors[p+self.suffix(trigger_idx)].latex_label[:-1] + "{sep_char}{{({n})}}$".format(sep_char=self.sep_char, n=trigger_idx+1)
                else:
                    self.joint_priors[p+self.suffix(trigger_idx)].latex_label += self.suffix(trigger_idx)
        self.joint_priors = bilby.core.prior.PriorDict(self.joint_priors)

        self.joint_search_parameter_keys = [k for k in list(self.joint_priors.keys()) if type(self.joint_priors[k]) != bilby.core.prior.Constraint]


    def generate_joint_posterior_samples(self, samples, n_mcmc=5000):
        """
            The basic idea of this MCMC (actually Metropolis-Hastings) algorithm is that we first randomly draw one 
            row (or equivalently an index) [prior: 1/N_row] from the base/0th trigger

            Conditioned on this row, draw a sample of (theta_ind^2)
            Accept this draw if a \def p(x_new)/p(x_old) >= 1
            Otherwise accept this draw with a probabiltiy of a
        """
        # Initialize the joint likelihood
        joint_likelihood = LensingJointLikelihoodWithWaveformCache(self.single_trigger_likelihoods, sep_char=self.sep_char, suffix=self.suffix)

        # Initialize the chain
        mcmc_chain = []
        log_posteriors = []

        def propose_sample():
            proposed_idx = np.random.randint(0, len(samples))
            proposed_sample = samples.iloc[proposed_idx].to_dict()
            joint_likelihood.parameters.update(proposed_sample)
            proposed_sample_log_L = joint_likelihood.log_likelihood()
            proposed_sample_log_prior = self.joint_priors.ln_prob(proposed_sample)

            return proposed_idx, proposed_sample, proposed_sample_log_L, proposed_sample_log_prior

        # Loop until n_mcmc of samples are generated
        for _ in tqdm.tqdm(range(n_mcmc)):
            p_idx, p_sample, p_log_L, p_log_prior = propose_sample()
            p_log_posterior = p_log_L+p_log_prior

            # Metropolis algorithm
            if len(mcmc_chain) == 0:
                mcmc_chain.append(p_idx)
                log_posteriors.append(p_log_posterior)
                continue

            acceptance_prob = np.amin([1, np.exp(p_log_posterior - log_posteriors[-1])])
            if np.random.random() <= acceptance_prob:
                # Accept
                mcmc_chain.append(p_idx)
            else:
                # Reject
                mcmc_chain.append(mcmc_chain[-1])

        return mcmc_chain

    def run(self):
        # FIXME Maybe make it deterministic instead?
        theta_to_evaluate = self.single_trigger_results[self.trigger_ids[0]].posterior.sample(self.n_posterior)

        log_Z_conditioned = []
        samples = [] # Not necessarily from posterior dist
        joint_posterior_samples = []

        logger = logging.getLogger(__prog__)
        if self.waveform_cache:
            logger.info("Using waveform caching")
 
        sampled_parameters = copy.deepcopy(self.independent_parameters)
        if self.likelihood_base_with_cache.time_marginalization:
            sampled_parameters.remove("geocent_time")
        if self.likelihood_base_with_cache.distance_marginalization:
            sampled_parameters.remove("luminosity_distance")

        # FIXME Check if there is anything to sample. If there is none, use the special method
        if sampled_parameters == ["image_type"]:
            logger.info("All parameters can be explicitly marginalized over. Disabling stochastic sampling")
            # Embarrassingly parallelized
            logger.info("Using {} CPU core(s) for rapid sampling".format(self.n_cores))
            
            with MultiPool(self.n_cores) as pool:
                outputs = pool.starmap(
                    sample_time_dist_marginalized,
                    tqdm.tqdm([[
                        theta_to_evaluate.iloc[i],
                        self.trigger_ids,
                        self.suffix,
                        self.likelihood_parameter_keys,
                        self.independent_parameters,
                        self.lensing_prior_dict,
                        self.likelihood_base,
                        self.single_trigger_likelihoods_with_cache,
                        self.waveform_cache,
                    ] for i in range(self.n_posterior)])
                )

            for i in range(len(outputs)):
                log_Z_conditioned.append(outputs[i][0])
                samples.append(outputs[i][1])
            logger.info("Rapid sampling completed")
        else:
            raise NotImplementedError("General sampling not implemented yet")

        samples = pd.DataFrame(samples)
        self.joint_parameter_keys = list(samples.columns)
        log_Z_conditioned = np.array(log_Z_conditioned)
        self.log_joint_evidence = compute_log_joint_evidence_from_log_conditional_evidence(self.single_trigger_results[self.trigger_ids[0]].log_evidence, log_Z_conditioned)
        self.log_joint_evidence_err, _ = bootstrap_uncertainty(log_Z_conditioned)
    
        if self.generate_posterior_samples:
            # Generate equal-weighted posterior samples
            logger.info("Generating joint posterior samples. This may take a moment")
            joint_posterior_samples = samples.iloc[self.generate_joint_posterior_samples(samples, n_mcmc=len(samples))]
        else:
            joint_posterior_samples = None

        log_joint_noise_evidence = np.sum([r.log_noise_evidence for r in self.single_trigger_results])
        samples["log_conditional_evidence"] = log_Z_conditioned

        # Save to file
        joint_result = bilby.core.result.Result(
            outdir=self.outdir,
            label=self.label,
            sampler="hanabi_rapid_analysis",
            search_parameter_keys=self.joint_search_parameter_keys,
            posterior=joint_posterior_samples,
            samples=samples,
            log_evidence=self.log_joint_evidence,
            log_evidence_err=self.log_joint_evidence_err,
            log_noise_evidence=log_joint_noise_evidence,
            log_bayes_factor=self.log_joint_evidence-log_joint_noise_evidence,
            priors=self.joint_priors
        )

        joint_result.save_to_file(outdir=self.outdir)

        return joint_result

def main():
    parser = create_rapid_analysis_parser(__prog__, __version__)
    args, unknown_args = bilby_pipe.utils.parse_args(bilby_pipe.utils.get_command_line_arguments(), parser)
    analysis = RapidAnalysisInput(args, unknown_args)
    analysis.run_analysis()
    sys.exit(0)
