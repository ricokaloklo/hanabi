#!/usr/bin/env python
import subprocess
import copy
import logging
import bilby_pipe
from parallel_bilby.utils import get_cli_args
from .parser import create_joint_generation_pbilby_parser
from .utils import write_complete_config_file as _write_complete_config_file
from .slurm_pbilby import setup_submit

from .._version import __version__
__prog__ = "hanabi_joint_generation_pbilby"

logger = logging.getLogger(__prog__)


class JointGenerationPBilbyInput(bilby_pipe.input.Input):
    def __init__(self, args):
        self.args = args

        self.outdir = args.outdir
        self.label = args.label
        # Read the rest of the supported arguments
        for name in dir(args):
            if not name.startswith("_"):
                setattr(self, name, getattr(args, name, None))

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"


def generate_single_trigger_pe_data_dump_files(joint_generation_input):
    data_dump_files = []

    for trigger_ini_file in joint_generation_input.trigger_ini_files:
        try:
            logger.info(f"Running parallel_bilby_generation for {trigger_ini_file}")
            subprocess.run(["parallel_bilby_generation {}".format(trigger_ini_file)], shell=True)
            data_dump_files.append("{data_dir}/{label}_data_dump.pickle".format(
                data_dir=joint_generation_input.data_directory,
                label=joint_generation_input.label,
            ))
        except:
            logger.error(f"Failed to generate parallel_bilby data dump file for {trigger_ini_file}")

    return data_dump_files

def write_complete_config_file(parser, args, inputs, prog):
    # Hack
    inputs_for_writing_config = copy.deepcopy(inputs)
    inputs_for_writing_config.request_cpus = 1
    inputs_for_writing_config.sampler_kwargs = "{}"
    inputs_for_writing_config.mpi_timing_interval = 0

    _write_complete_config_file(parser, args, inputs_for_writing_config, prog)


def main():
    cli_args = get_cli_args()
    parser = create_joint_generation_pbilby_parser(__prog__, __version__)
    args = parser.parse_args(args=cli_args)

    joint_generation_input = JointGenerationPBilbyInput(args)

    # Calling parallel_bilby_generation to generate the data dump files
    data_dump_files = generate_single_trigger_pe_data_dump_files(joint_generation_input)

    write_complete_config_file(parser, args, joint_generation_input, __prog__)

    # Generate bash file for slurm submission
    bash_file = setup_submit(data_dump_files, joint_generation_input, args)
    if args.submit:
        subprocess.run(["bash {}".format(bash_file)], shell=True)
    else:
        logger.info("Setup complete, now run:\n $ bash {}".format(bash_file))
