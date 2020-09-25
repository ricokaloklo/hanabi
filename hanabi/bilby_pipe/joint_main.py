import os
import logging

import bilby
import bilby_pipe
import bilby_pipe.main
from bilby_pipe.job_creation.bilby_pipe_dag_creator import get_parallel_list, create_overview
from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.nodes import MergeNode
from .analysis_node import JointAnalysisNode

# NOTE Importing the following will initialize a logger for bilby_pipe
import bilby_pipe.utils
# NOTE Importing the following will initialize a logger for hanabi_joint_pipe
from .utils import setup_logger

from .parser import create_joint_parser, print_unrecognized_arguments
from ._version import __version__

__prog__ = "hanabi_joint_pipe"

class JointMainInput(bilby_pipe.input.Input):
    def __init__(self, args, unknown_args):
        self.known_args = args
        self.unknown_args = unknown_args

        self.n_triggers = args.n_triggers
        self.trigger_ini_files = args.trigger_ini_files
        self.common_parameters = args.common_parameters
        self.lensing_prior_dict = args.lensing_prior_dict
        self.lensed_waveform_model = args.lensed_waveform_model

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"

        """
        These are the options that hanabi_joint_pipe
        should also recognize, copied from bilby_pipe
        """
        self.submit = args.submit
        self.condor_job_priority = args.condor_job_priority
        self.online_pe = args.online_pe
        self.create_plots = args.create_plots
        self.singularity_image = args.singularity_image
        self.create_summary = args.create_summary

        self.outdir = args.outdir
        self.label = args.label
        self.log_directory = args.log_directory
        self.accounting = args.accounting
        self.sampler = args.sampler
        self.n_parallel = args.n_parallel
        self.transfer_files = args.transfer_files
        self.osg = args.osg

        self.webdir = args.webdir
        self.email = args.email
        self.notification = args.notification
        self.existing_dir = args.existing_dir

        self.scheduler = args.scheduler
        self.scheduler_args = args.scheduler_args
        self.scheduler_module = args.scheduler_module
        self.scheduler_env = args.scheduler_env
        self.scheduler_analysis_time = args.scheduler_analysis_time

        self.run_local = args.local
        self.local_generation = args.local_generation
        self.local_plot = args.local_plot

        self.request_memory = args.request_memory
        self.request_memory_generation = args.request_memory_generation
        self.request_cpus = args.request_cpus
        self.sampler_kwargs = args.sampler_kwargs

        if self.create_plots:
            for plot_attr in [
                "calibration",
                "corner",
                "marginal",
                "skymap",
                "waveform",
                "format",
            ]:
                attr = f"plot_{plot_attr}"
                setattr(self, attr, getattr(args, attr))

        self.postprocessing_executable = args.postprocessing_executable
        self.postprocessing_arguments = args.postprocessing_arguments
        self.single_postprocessing_executable = args.single_postprocessing_executable
        self.single_postprocessing_arguments = args.single_postprocessing_arguments

        self.summarypages_arguments = args.summarypages_arguments

        self.extra_lines = []
        self.requirements = []

        # Turn off automatic submission
        if self.submit:
            logger = logging.getLogger(__prog__)
            logger.info(f"Turning off automatic submission")
            self.submit = False

    # The following lines of code are also modified from bilby_pipe
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

    # PlotNode
    plot_nodes_list = []
    for merged_node in merged_node_list:
        if joint_main_input.create_plots:
            plot_nodes_list.append(PlotNode(joint_main_input, merged_node, dag=dag))
        if joint_main_input.single_postprocessing_executable:
            PostProcessSingleResultsNode(joint_main_input, merged_node, dag=dag)

    if joint_main_input.create_summary:
        PESummaryNode(joint_main_input, merged_node_list, generation_node_list, dag=dag)
    if joint_main_input.postprocessing_executable is not None:
        PostProcessAllResultsNode(joint_main_input, merged_node_list, dag)

    dag.build()

def write_complete_config_file(parser, args, inputs):
    # Also copied from bilby_pipe
    args_dict = vars(args).copy()
    for key, val in args_dict.items():
        if key == "label":
            continue
        if isinstance(val, str):
            if os.path.isfile(val) or os.path.isdir(val):
                setattr(args, key, os.path.abspath(val))
        if isinstance(val, list):
            if isinstance(val[0], str):
                setattr(args, key, f"[{', '.join(val)}]")
    args.sampler_kwargs = str(inputs.sampler_kwargs)

    logger = logging.getLogger(__prog__)
    logger.info(f"Writing the complete config ini file to {inputs.complete_ini_file}")

    parser.write_to_file(
        filename=inputs.complete_ini_file,
        args=args,
        overwrite=False,
        include_description=False,
    )

def main():
    """
    Top level interface for hanabi_joint_pipe
    """
    parser = create_joint_parser(__prog__, __version__)
    args, unknown_args = bilby_pipe.utils.parse_args(bilby_pipe.utils.get_command_line_arguments(), parser)

    # Initialize our own logger
    joint_main_logger = logging.getLogger(__prog__)
    # Log the version of our package
    joint_main_logger.info(f"Running {__prog__} version: {__version__}")

    joint_main_input = JointMainInput(args, unknown_args)
    # Write the dags for single trigger analysis
    single_trigger_pe_inputs = generate_single_trigger_pe_inputs(joint_main_input, write_dag=True)

    # Write the complete config ini file
    write_complete_config_file(parser, args, joint_main_input)

    # Generate the dag for the joint analysis
    generate_dag(joint_main_input, single_trigger_pe_inputs)

    print_unrecognized_arguments(unknown_args, joint_main_logger)

