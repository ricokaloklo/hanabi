#!/usr/bin/env python
""" Script to perform joint data analysis """
import os
import sys
import signal
import logging
from importlib import import_module

import numpy as np
import bilby
import bilby.core.prior
import bilby_pipe
import bilby_pipe.data_analysis
from bilby_pipe.utils import CHECKPOINT_EXIT_CODE

# Lensed waveform source model
from hanabi.lensing.waveforms import *

# Lensing likelihood
import hanabi.lensing.likelihood
from .utils import setup_logger
from .parser import create_joint_parser

from ._version import __version__
__prog__ = "hanabi_joint_analysis"

def sighandler(signum, frame):
    logger = logging.getLogger(__prog__)
    logger.info("Performing periodic eviction")
    sys.exit(CHECKPOINT_EXIT_CODE)


class SingleTriggerDataAnalysisInput(bilby_pipe.data_analysis.DataAnalysisInput):
    def run_sampler(self):
        logger = logging.getLogger(__prog__)
        msg = [bilby_pipe.utils.tcolors.WARNING, f"The run_sampler() function for SingleTriggerDataAnalysisInput is being invoked", bilby_pipe.utils.tcolors.END]
        logger.warning(" ".join(msg))


class JointDataAnalysisInput(bilby_pipe.input.Input):
    def __init__(self, args, unknown_args, test=False):
        """
        Initalize multiple SingleTriggerDataAnalysisInput
        """
        self.n_triggers = args.n_triggers
        # NOTE The ini files passed should be the complete ones
        self.trigger_ini_files = args.trigger_ini_files
        self.common_parameters = args.common_parameters
        self.lensing_prior_dict = args.lensing_prior_dict
        self.data_dump_files = args.data_dump_files
        self.lensed_waveform_model = args.lensed_waveform_model
        # Parse the lensing prior dict
        self.parse_lensing_prior_dict()

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # Choices for running
        self.sampler = args.sampler
        self.sampler_kwargs = args.sampler_kwargs
        self.sampling_seed = args.sampling_seed

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"
        assert self.n_triggers == len(self.data_dump_files), "n_triggers does not match with the number of data dump files given"
        assert self.n_triggers == len([k for k in self.lensing_prior_dict.keys() if 'image_type' in k]), "n_triggers does not match with the number of image_type priors given"
        assert self.n_triggers == len([k for k in self.lensing_prior_dict.keys() if 'magnification' in k]), "n_triggers does not match with the number of magnification priors given"

        # FIXME Make sure no one will use the ^ sign to name actual parameter
        self.sep_char = "^"
        self.suffix = lambda trigger_idx: "{}({})".format(self.sep_char, trigger_idx + 1)

        # Initialize multiple SingleTriggerDataAnalysisInput objects
        self.initialize_single_trigger_data_analysis_inputs()

    @property
    def lensed_waveform_model(self):
        return self._lensed_waveform_model

    @lensed_waveform_model.setter
    def lensed_waveform_model(self, lensed_waveform_model):
        # A list of built-in lensed_waveform_model string
        built_in_lensed_waveform_model_dict = {
            "strongly_lensed_BBH_waveform": strongly_lensed_BBH_waveform
        }

        logger = logging.getLogger(__prog__)
        logger.info(f"Using the lensed waveform model {lensed_waveform_model}")

        if lensed_waveform_model in built_in_lensed_waveform_model_dict.keys():
            self._lensed_waveform_model = built_in_lensed_waveform_model_dict[lensed_waveform_model]
        elif "." in lensed_waveform_model:
            split_model = lensed_waveform_model.split(".")
            module = ".".join(split_model[:-1])
            func = split_model[-1]
            self._lensed_waveform_model = getattr(import_module(module), func)
        else:
            raise FileNotFoundError(
                f"No lensed waveform model {lensed_waveform_model} found."
            )

    @property
    def common_parameters(self):
        return self._common_parameters

    @common_parameters.setter
    def common_parameters(self, common_parameters):
        if common_parameters is not None:
            self._common_parameters = common_parameters
        else:
            self._common_parameters = []

    @property
    def sampling_seed(self):
        return self._samplng_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._samplng_seed = sampling_seed
        np.random.seed(sampling_seed)
        logger = logging.getLogger(__prog__)
        logger.info(f"Sampling seed set to {sampling_seed}")

        if self.sampler == "cpnest":
            self.sampler_kwargs["seed"] = self.sampler_kwargs.get(
                "seed", self._samplng_seed
            )

    @property
    def result_directory(self):
        result_dir = os.path.join(self.outdir, "result")
        return os.path.relpath(result_dir)

    def initialize_single_trigger_data_analysis_inputs(self):
        self.single_trigger_data_analysis_inputs = []
        self.single_trigger_likelihoods = []
        self.single_trigger_priors = []

        logger = logging.getLogger(__prog__)

        for complete_trigger_ini_file, data_dump_file in zip(self.trigger_ini_files, self.data_dump_files):
            logger.info(f"Parsing config ini file {complete_trigger_ini_file}")
            args, unknown_args = bilby_pipe.utils.parse_args([complete_trigger_ini_file, "--data-dump-file", data_dump_file], bilby_pipe.data_analysis.create_analysis_parser())
            single_trigger_analysis = SingleTriggerDataAnalysisInput(args, unknown_args)
            self.single_trigger_data_analysis_inputs.append(single_trigger_analysis)

            likelihood, priors = single_trigger_analysis.get_likelihood_and_priors()
            self.single_trigger_likelihoods.append(likelihood)
            self.single_trigger_priors.append(priors)

    def parse_lensing_prior_dict(self):
        # Pre-process/convert the prior dict string
        lensing_prior_dict = bilby_pipe.utils.convert_prior_string_input(self.lensing_prior_dict)
        lensing_prior_dict = {
            bilby_pipe.input.Input._convert_prior_dict_key(key): val for key, val in lensing_prior_dict.items()
        }

        self.lensing_prior_dict = bilby.core.prior.PriorDict(lensing_prior_dict)

    def construct_full_prior_dict(self):
        full_prior_dict = {}
        logger = logging.getLogger(__prog__)

        # Add vanilla CBC, calibration and lensing parameters to full prior dict
        for trigger_idx, (likelihood, cbc_priors) in enumerate(zip(self.single_trigger_likelihoods, self.single_trigger_priors)):
            full_parameters = likelihood.priors.keys()
            suffix = self.suffix(trigger_idx)
            
            for param in full_parameters:
                if param not in self.common_parameters:
                    # Add indepedent parameters to the full prior dict
                    full_prior_dict[param + suffix] = likelihood.priors[param]
                    # Rename the prior
                    full_prior_dict[param + suffix].name = full_prior_dict[param + suffix].name + suffix
                if param in self.common_parameters:
                    if param in full_prior_dict.keys():
                        # Already added to the full prior dict, check consistency
                        assert full_prior_dict[param] == likelihood.priors[param]
                    else:
                        # Add common parameters to the full prior dict
                        full_prior_dict[param] = likelihood.priors[param]

            # Add lensing priors
            full_prior_dict["image_type" + suffix] = self.lensing_prior_dict["image_type" + suffix]
            # Check if both relative_magnification and absolute_magnification are given
            if "relative_magnification" in self.lensing_prior_dict.keys() and "absolute_magnification" in self.lensing_prior_dict.keys():
                raise ValueError("Cannot give relative_magnification and absolute_magnification at the same time")

            try:
                full_prior_dict["relative_magnification" + suffix] = self.lensing_prior_dict["relative_magnification" + suffix]
            except:
                full_prior_dict["absolute_magnification" + suffix] = self.lensing_prior_dict["absolute_magnification" + suffix]
            
        return full_prior_dict

    def get_likelihood_and_priors(self):
        """
        Return JointLikelihood and the full prior with lensing parameters
        """
        # Construct the full_prior_dict
        priors = self.construct_full_prior_dict()

        # Construct the LensingJointLikelihood object
        likelihood = hanabi.lensing.likelihood.LensingJointLikelihood(self.single_trigger_likelihoods, self.lensed_waveform_model, sep_char=self.sep_char, suffix=self.suffix)

        return likelihood, priors

    def run_sampler(self):
        if self.scheduler.lower() == "condor":
            signal.signal(signal.SIGALRM, handler=sighandler)
            signal.alarm(self.periodic_restart_time)

        likelihood, priors = self.get_likelihood_and_priors()

        self.result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=self.sampler,
            label=self.label,
            outdir=self.result_directory,
            exit_code=CHECKPOINT_EXIT_CODE,
            **self.sampler_kwargs
        )


def create_joint_analysis_parser():
    parser = create_joint_parser(__prog__, __version__)

    # Add new options
    parser.add(
        "--data-dump-files",
        action="append",
        help=(
            "A list of data dump files for each trigger, "
            "specified either by `data-dump-files=[FILE1, FILE2]` or "
            "as command-line arguments by `--data-dump-files FILE1 --data-dump-files FILE2`"
        )
    )

    return parser

def main():
    args, unknown_args = bilby_pipe.utils.parse_args(sys.argv[1:], create_joint_analysis_parser())
    analysis = JointDataAnalysisInput(args, unknown_args)
    analysis.run_sampler()
    sys.exit(0)