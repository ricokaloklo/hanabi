import bilby
import bilby_pipe
import bilby_pipe.main
import logging

# NOTE Importing the following will initialize a logger for bilby_pipe
import bilby_pipe.utils
# NOTE Importing the following will initialize a logger for hanabi_joint_pipe
from .utils import setup_logger

from .parser import create_joint_parser, print_unrecognized_arguments
from ._version import __version__

__prog__ = "hanabi_joint_pipe"

class JointMainInput(bilby_pipe.input.Input):
    def __init__(self, args, unknown_args):
        """
        Read the number of events N, then initialize multiple bilby_pipe.Input
        """
        self.n_triggers = args.n_triggers
        self.trigger_ini_files = args.trigger_ini_files
        self.common_parameters = args.common_parameters
        self.lensing_prior_dict = args.lensing_prior_dict

        # Sanity check
        assert self.n_triggers == len(self.trigger_ini_files), "n_triggers does not match with the number of config files"

    def generate_single_trigger_dags(self):
        _ = self.generate_single_trigger_pe_inputs(write_dag=True)

    def generate_single_trigger_pe_inputs(self, write_dag=False):
        single_trigger_pe_inputs = []
        logger = logging.getLogger(__prog__)
        for trigger_ini_file in self.trigger_ini_files:
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

    def generate_dag(self):
        pass

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
    joint_main_input.generate_single_trigger_dags()

    # Write the dag for the joint analysis
    joint_main_input.generate_dag()

    print_unrecognized_arguments(unknown_args, joint_main_logger)

