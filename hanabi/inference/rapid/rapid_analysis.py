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

from .parser import create_rapid_analysis_parser
from .utils import load_run_from_bilby, load_run_from_pbilby
from .utils import _dist_marg_lookup_table_filename_template
from .likelihood import SingleLikelihoodWithTransformableWaveformCache
from ..utils import ParameterSuffix

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
            self.trigger_combo = self.trigger_ids
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

        for idx, (trigger_ini_file, data_dump_file, result_file, PE_program) in enumerate(zip(self.trigger_ini_files, self.data_dump_files, self.result_files, self.inference_software)):
            if PE_program == "bilby":
                single_trigger_likelihood, priors, single_trigger_result = \
                    load_run_from_bilby(
                        result_file,
                        data_dump_file,
                        trigger_ini_file,
                    )                
            elif PE_program == "parallel_bilby":
                single_trigger_likelihood, priors, single_trigger_result = \
                    load_run_from_pbilby(
                        result_file,
                        data_dump_file,
                    )
            else:
                raise ValueError("Cannot recognize/does not support the inference software {}".format(PE_program))

            single_trigger_likelihood.priors = priors

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

    def run_analysis(self):
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
            )

            logger = logging.getLogger(__prog__)
            logger.info("Running inference on {} conditioned on {}".format(
                " ,".join([self.single_trigger_results[i].label for i in combo[1:]]),
                self.single_trigger_results[combo[0]].label
            ))
            inference.run()

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

        self.sep_char = sep_char
        if suffix is None:
            suffix = ParameterSuffix(self.sep_char)
        self.suffix = suffix

        self.likelihood_base = self.single_trigger_likelihoods[self.trigger_ids[0]] # Condition on this trigger
        # NOTE This assumes that all the likelihood functions take the same input parameters
        self.likelihood_parameter_keys = \
            list(self.likelihood_base.priors.sample().keys())
        self.independent_parameters = [p for p in self.likelihood_parameter_keys if p not in self.common_parameters]

        # Reconstruct the effective joint prior
        # The priors for the common parameters are from the base trigger
        self.joint_priors = {k: self.single_trigger_priors[self.trigger_ids[0]][k] for k in self.common_parameters}
        for trigger_idx in self.trigger_ids:
            for p in self.independent_parameters:
                self.joint_priors[p+self.suffix(trigger_idx)] = copy.deepcopy(self.single_trigger_priors[trigger_idx][p])
                # Rename the parameter
                self.joint_priors[p+self.suffix(trigger_idx)].name += self.suffix(trigger_idx)
                # LaTeX label
                if self.joint_priors[p+self.suffix(trigger_idx)].latex_label.endswith("$"):
                    self.joint_priors[p+self.suffix(trigger_idx)].latex_label = \
                        self.joint_priors[p+self.suffix(trigger_idx)].latex_label[:-1] + "{sep_char}{{({n})}}$".format(sep_char=self.sep_char, n=trigger_idx+1)
                else:
                    self.joint_priors[p+self.suffix(trigger_idx)].latex_label += self.suffix(trigger_idx)

    def sample_all_marginalized(self, theta):
        # Evaluate likelihood_base once to generate the waveform
        theta_dict = {p: theta[p] for p in self.likelihood_parameter_keys}

        if self.waveform_cache:
            self.likelihood_base.parameters.update(theta_dict)
            self.likelihood_base.log_likelihood_ratio()

        log_evidence = 0.
        posterior_to_add = {}

        for trigger_idx in self.trigger_ids[1:]:
            conditioned_likelihood = self.single_trigger_likelihoods_with_cache[trigger_idx]
            conditioned_likelihood.parameters.update(theta_dict)

            if self.waveform_cache:
                # Assign waveform cache (so that we do not need to re-evaluate waveform)
                conditioned_likelihood.initialize_cache(self.likelihood_base.waveform_generator._cache, theta_dict)

            conditioned_likelihood.parameters.update({
                "geocent_time": float(conditioned_likelihood.interferometers.start_time)
            })

            # For each image type, compute the logL
            image_types = [1.0, 2.0, 3.0]
            log_priors = []
            log_Ls = []            

            for image_type in image_types:
                log_priors.append(
                    self.lensing_prior_dict["image_type"+self.suffix(trigger_idx)].ln_prob(image_type)
                )
                conditioned_likelihood.parameters.update({'image_type': image_type})
                log_Ls.append(conditioned_likelihood.log_likelihood())

            log_priors = np.array(log_priors)
            log_Ls = np.array(log_Ls)

            log_evidence += logsumexp(log_Ls, b=np.exp(log_priors))

            # Draw one posterior sample for each image type, weighted by log posterior
            drawn_image_type = np.random.choice(image_types, p=np.exp(log_Ls + log_priors - log_evidence))
            conditioned_likelihood.parameters.update({'image_type': drawn_image_type})
            drawn_sample = conditioned_likelihood.generate_posterior_sample_from_marginalized_likelihood()

            # Rename independent parameters
            for k in self.independent_parameters:
                drawn_sample[k+self.suffix(trigger_idx)] = drawn_sample.pop(k)
            posterior_to_add.update(drawn_sample)

        for k in self.independent_parameters:
            posterior_to_add[k+self.suffix(self.trigger_ids[0])] = theta_dict[k] 

        return log_evidence, posterior_to_add

    def run(self):
        # FIXME Maybe make it deterministic instead?
        theta_to_evaluate = self.single_trigger_results[self.trigger_ids[0]].posterior.sample(self.n_posterior)

        log_Z_conditioned = []
        joint_posterior_samples = []

        logger = logging.getLogger(__prog__)
        if self.waveform_cache:
            logger.info("Using waveform caching")
 
        # FIXME Check if there is anything to sample. If there is none, use the special method
        self.sample = self.sample_all_marginalized
        # All parameters are marginalized. Parallelization does not worth the overhead and extra memory usage
        logger.info("Using 1 CPU core only, despite --request-cpus was set to {}".format(self.n_cores))

        for i in tqdm.tqdm(range(self.n_posterior)):
            log_ev, pos = self.sample(theta_to_evaluate.iloc[i])
            log_Z_conditioned.append(log_ev)
            joint_posterior_samples.append(pos)

        log_joint_evidence = self.single_trigger_results[self.trigger_ids[0]].log_evidence + logsumexp(np.array(log_Z_conditioned)) - np.log(self.n_posterior)
        joint_posterior_samples = pd.DataFrame(joint_posterior_samples)
        joint_posterior_samples["log_conditional_evidence"] = log_Z_conditioned

        # Save to file
        joint_result = bilby.core.result.Result(
            outdir=self.outdir,
            label="{label}_conditioned_on_{base_label}".format(label=self.label, base_label=self.single_trigger_results[self.trigger_ids[0]].label),
            sampler="hanabi_rapid_analysis",
            search_parameter_keys=[k for k in list(self.joint_priors.keys()) if type(self.joint_priors[k]) != bilby.core.prior.Constraint],
            posterior=joint_posterior_samples,
            log_evidence=log_joint_evidence,
            priors=self.joint_priors
        )

        joint_result.save_to_file(outdir=self.outdir)

def main():
    parser = create_rapid_analysis_parser(__prog__, __version__)
    args, unknown_args = bilby_pipe.utils.parse_args(bilby_pipe.utils.get_command_line_arguments(), parser)
    analysis = RapidAnalysisInput(args, unknown_args)
    analysis.run_analysis()
    sys.exit(0)
