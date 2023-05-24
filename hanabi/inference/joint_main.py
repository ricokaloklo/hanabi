import os
import logging

import bilby
import bilby_pipe
import bilby_pipe.main
from bilby_pipe.job_creation.bilby_pipe_dag_creator import get_parallel_list, create_overview
from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.nodes import MergeNode, PostProcessSingleResultsNode
from bilby_pipe.utils import BilbyPipeError
from .analysis_node import JointAnalysisNode

# NOTE Importing the following will initialize a logger for bilby_pipe
import bilby_pipe.utils
# NOTE Importing the following will initialize a logger for hanabi_joint_pipe
from .utils import (
    setup_logger,
    write_complete_config_file,
    turn_off_forbidden_option,
)

from .parser import create_joint_main_parser, print_unrecognized_arguments
from .utils import get_version_information
__version__ = get_version_information()

__prog__ = "hanabi_joint_pipe"

class JointMainInput(bilby_pipe.input.Input):
    def __init__(self, args, unknown_args):
        self.known_args = args
        self.unknown_args = unknown_args

        self.outdir = args.outdir
        self.label = args.label
        self.run_local = args.local
        # Read the rest of the supported arguments
        for name in dir(args):
            if not name.startswith("_"):
                setattr(self, name, getattr(args, name, None))

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"

        self.extra_lines = []
        self.requirements = []

    # The following lines of code are also modified from bilby_pipe
    @property
    def ini(self):
        return self._ini

    @ini.setter
    def ini(self, ini):
        if os.path.isfile(ini) is False:
            raise FileNotFoundError(f"No ini file {ini} found")
        self._ini = os.path.relpath(ini)

    @property
    def notification(self):
        return self._notification

    @notification.setter
    def notification(self, notification):
        valid_settings = ["Always", "Complete", "Error", "Never"]
        if notification in valid_settings:
            self._notification = notification
        else:
            raise BilbyPipeError(
                "'{}' is not a valid notification setting. "
                "Valid settings are {}.".format(notification, valid_settings)
            )

    @property
    def initialdir(self):
        return os.getcwd()

    @property
    def request_disk(self):
        return self._request_disk

    @request_disk.setter
    def request_disk(self, request_disk):
        self._request_disk = f"{request_disk}GB"
        self._request_disk_in_GB = float(request_disk)
        logger = logging.getLogger(__prog__)
        logger.debug(f"Setting analysis request_disk={self._request_disk}")
        self._request_disk = f"{request_disk}GB"

    @property
    def request_memory(self):
        return self._request_memory

    @request_memory.setter
    def request_memory(self, request_memory):
        logger = logging.getLogger(__prog__)
        logger.info(f"Setting analysis request_memory={request_memory}GB")
        self._request_memory = f"{request_memory} GB"

    @property
    def request_memory_generation(self):
        return self._request_memory_generation

    @request_memory_generation.setter
    def request_memory_generation(self, request_memory_generation):
        logger = logging.getLogger(__prog__)
        logger.info(f"Setting request_memory_generation={request_memory_generation}GB")
        self._request_memory_generation = f"{request_memory_generation} GB"

    @property
    def request_cpus(self):
        return self._request_cpus

    @request_cpus.setter
    def request_cpus(self, request_cpus):
        logger = logging.getLogger(__prog__)
        logger.info(f"Setting analysis request_cpus = {request_cpus}")
        self._request_cpus = request_cpus


def generate_single_trigger_pe_inputs(joint_main_input, write_dag=False):
    single_trigger_pe_inputs = []
    logger = logging.getLogger(__prog__)
    for trigger_ini_file in joint_main_input.trigger_ini_files:
        logger.info(f"Parsing config ini file {trigger_ini_file}")
        bilby_pipe_parser = bilby_pipe.main.create_parser(top_level=True)
        # NOTE We should probably figure out a mechanism to make sure that the data dump files were generated prior to joint analysis
        args, unknown_args = bilby_pipe.utils.parse_args([trigger_ini_file], bilby_pipe_parser)
        main_input = bilby_pipe.main.MainInput(args, unknown_args)

        turn_off_forbidden_option(main_input, "submit", __prog__)
        turn_off_forbidden_option(args, "submit", __prog__)
        turn_off_forbidden_option(args, "phase_marginalization", __prog__)
        turn_off_forbidden_option(args, "time_marginalization", __prog__)

        if write_dag:
            bilby_pipe.main.write_complete_config_file(bilby_pipe_parser, args, main_input)
            bilby_pipe.main.generate_dag(main_input)
    
        single_trigger_pe_inputs.append(main_input)

    return single_trigger_pe_inputs


def generate_dag(joint_main_input, single_trigger_pe_inputs):
    dag = Dag(joint_main_input)
    parallel_list = get_parallel_list(joint_main_input)

    # NOTE There is no GenerationNode as all data generations are done in single trigger
    merged_node_list = []
    all_parallel_node_list = []
    parallel_node_list = []

    # (Parallel) AnalysisNode and MergeNode

    # Suppress bilby_pipe logging
    try:
        bilby_pipe_logger = logging.getLogger("bilby_pipe")
        old_level = bilby_pipe_logger.level
        bilby_pipe_logger.setLevel(logging.CRITICAL)
    except:
        pass

    for parallel_idx in parallel_list:
        analysis_node = JointAnalysisNode(
            joint_main_input,
            single_trigger_pe_inputs,
            parallel_idx=parallel_idx,
            dag=dag
        )
        parallel_node_list.append(analysis_node)
        all_parallel_node_list.append(analysis_node)

    # Resume bilby_pipe logging
    try:
        bilby_pipe_logger.setLevel(old_level)
    except:
        pass

    if len(parallel_node_list) == 1:
        merged_node_list.append(analysis_node)
    else:
        merge_node = MergeNode(
                        inputs=joint_main_input,
                        parallel_node_list=parallel_node_list,
                        detectors=None,
                        dag=dag,
        )
        merged_node_list.append(merge_node)

    for merged_node in merged_node_list:
        if joint_main_input.single_postprocessing_executable:
            # Support PostProcessSingleResultsNode
            PostProcessSingleResultsNode(joint_main_input, merged_node, dag=dag)

    dag.build()


def main():
    """
    Top level interface for hanabi_joint_pipe
    """
    parser = create_joint_main_parser(__prog__, __version__)
    args, unknown_args = bilby_pipe.utils.parse_args(bilby_pipe.utils.get_command_line_arguments(), parser)

    # Initialize our own logger
    joint_main_logger = logging.getLogger(__prog__)
    # Log the version of our package
    joint_main_logger.info(f"Running {__prog__} version: {__version__}")

    joint_main_input = JointMainInput(args, unknown_args)
    # Write the dags for single trigger analysis
    single_trigger_pe_inputs = generate_single_trigger_pe_inputs(joint_main_input, write_dag=True)

    # Write the complete config ini file
    write_complete_config_file(parser, args, joint_main_input, __prog__)

    # Generate the dag for the joint analysis
    generate_dag(joint_main_input, single_trigger_pe_inputs)

    print_unrecognized_arguments(unknown_args, joint_main_logger)

