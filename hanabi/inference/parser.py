import bilby
import parallel_bilby
import bilby_pipe
import bilby_pipe.bilbyargparser
from bilby_pipe.parser import StoreBoolean
from bilby_pipe.utils import nonestr, nonefloat, noneint
import argparse
import logging
from parallel_bilby.parser.generation import (
    _add_slurm_settings_to_parser,
    _create_reduced_bilby_pipe_parser,
)
from parallel_bilby.parser.shared import (
    _add_dynesty_settings_to_parser,
    _add_misc_settings_to_parser,
)

def purge_empty_argument_group(parser):
    non_empty_action_groups = []
    non_empty_mutually_exclusive_groups = []

    try:
        # Purge _action_groups
        for action_group in parser._action_groups:
            if action_group._group_actions != []:
                non_empty_action_groups.append(action_group)
        # Purge _mutually_exclusive_groups
        for action_group in parser._mutually_exclusive_groups:
            if action_group._group_actions != []:
                non_empty_mutually_exclusive_groups.append(action_group)
    except:
        pass

    parser._action_groups = non_empty_action_groups
    parser._mutually_exclusive_groups = non_empty_mutually_exclusive_groups


def remove_arguments_from_parser(parser, args, prog):
    logger = logging.getLogger(prog)

    for arg in args:
        for action in parser._actions:
            if action.dest == arg.replace("-", "_"):
                try:
                    parser._handle_conflict_resolve(None, [("--" + arg, action)])
                except ValueError as e:
                    logger.warning("Error removing {}: {}".format(arg, e))
        logger.debug(
            "Request to remove arg {} from bilby_pipe args, but arg not found".format(arg)
        )

    purge_empty_argument_group(parser)


def keep_arguments_from_parser(parser, args, prog):
    original_args = [action.dest.replace("_", "-") for action in parser._actions]
    args_to_remove = list(set(original_args) - set(args))

    remove_arguments_from_parser(parser, args_to_remove, prog)


def _create_base_parser(prog, prog_version):
    base_parser = argparse.ArgumentParser(prog, add_help=False)
    base_parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={prog_version}\nbilby={bilby.__version__}\nbilby_pipe={bilby_pipe.__version__}\nparallel_bilby={parallel_bilby.__version__}"
    )

    return base_parser

def _list_arguments_in_group_by_title(parser, group_tilte):
    # Go through the list of _action_groups in parser, look for group with the given title
    # NOTE The first element in _action_groups is for positional arguments
    # and the second element is for options that are not in any particular group
    # such as --version and --help
    arguments_to_remove = []

    for group in parser._action_groups[2:]:
        if group.title == group_tilte:
            for action in group._group_actions:
                arguments_to_remove.append(action.option_strings[-1][2:])

    return arguments_to_remove

def _remove_arguments_from_bilby_pipe_parser_for_hanabi(bilby_pipe_parser, prog):
    bilby_pipe_arguments_remove = [
        'version',
    ]

    # These arguments are removed from hanabi_joint_pipe because they should be handled for each trigger
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Calibration arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Data generation arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Detector arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Injection arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Likelihood arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Prior arguments')
    bilby_pipe_arguments_remove += _list_arguments_in_group_by_title(bilby_pipe_parser, 'Waveform arguments')

    remove_arguments_from_parser(bilby_pipe_parser, bilby_pipe_arguments_remove, prog)

    return bilby_pipe_parser

def _add_hanabi_settings_to_parser(parser):
    # Make hanabi_joint_analysis to retry for the data generation jobs to complete
    parser.add(
        "--retry-for-data-generation",
        type=int,
        default=0,
        help=(
            "Retry an analysis node every N minute to wait for the data generation jobs "
            "from single trigger to complete. Default is to not retry"
        )
    )

    # Dealing with multiple inputs
    joint_input_parser = parser.add_argument_group(title="Joint input arguments", description="Specify multiple inputs")
    joint_input_parser.add(
        "--n-triggers",
        type=int,
        help="Number of triggers to be analyzed jointly"
    )

    joint_input_parser.add(
        "--trigger-ini-files",
        action="append",
        help=(
            "A list of configuration ini files for each trigger to be jointly analyzed, "
            "specified either by `trigger-ini-file=[PATH_1, PATH2]` or "
            "as command-line arguments by `--trigger-ini-files PATH1 --trigger-ini-files PATH2`"
        )
    )

    # Dealing with lensing speific prior
    lensing_prior_parser = parser.add_argument_group(title="Lensing prior argument", description="Specify the prior settings for lensing analysis")
    lensing_prior_parser.add(
        "--common-parameters",
        action="append",
        help=(
            "A list of common parameters that are imposed to take the same value across different triggers, "
            "specified either by `common-parameters=[chirp_mass, mass_ratio]` or "
            "as command-line arguments by `--common-parameters chirp_mass --common-parameters mass_ratio`"
        )
    )
    lensing_prior_parser.add(
        "--lensing-prior-dict",
        type=nonestr,
        default=None,
        help=(
            "A dictionary of prior for lensing magnification and image type. "
            "Specified by lensing-prior-dict={relative-magnification^(1) = 1, relative-magnification^(2) = PowerLaw(...)} for relative magnification, or "
            "lensing-prior-dict={absolute-magnification^(1) = PowerLaw(...), absolute-magnification^(2) = PowerLaw(...)} for absolute magnification"
        )
    )

    # Lensed waveform arguments
    lensed_waveform_parser = parser.add_argument_group(
        title="Waveform arguments", description="Setting for the waveform generator for lensed signals"
    )

    lensed_waveform_parser.add(
        "--waveform-cache",
        default=False,
        action=StoreBoolean,
        help="Enable waveform caching",
    )

    return parser

def create_joint_main_parser(prog, prog_version):
    base_parser = _create_base_parser(prog, prog_version)
    bilby_pipe_parser = _remove_arguments_from_bilby_pipe_parser_for_hanabi(
        bilby_pipe.parser.create_parser(), prog
    )

    # Remove --local-generation
    remove_arguments_from_parser(bilby_pipe_parser, ['local-generation'], prog)
    # Add our custom --local-generation back
    bilby_pipe_parser.add(
        "--local-generation",
        action="store_true",
        help="Run the data generation job locally, at the runtime of this program",
    )

    joint_main_parser = bilby_pipe.bilbyargparser.BilbyArgParser(
        prog=prog,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        add_help=False,
        parents=[base_parser, bilby_pipe_parser]
    )

    joint_main_parser = _add_hanabi_settings_to_parser(joint_main_parser)
    purge_empty_argument_group(joint_main_parser)

    return joint_main_parser

def create_joint_analysis_parser(prog, prog_version):
    parser = create_joint_main_parser(prog, prog_version)

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

def create_joint_generation_pbilby_parser(prog, prog_version):
    base_parser = _create_base_parser(prog, prog_version)
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    bilby_pipe_parser = _remove_arguments_from_bilby_pipe_parser_for_hanabi(
        bilby_pipe_parser, prog
    )

    joint_generation_pbilby_parser = bilby_pipe.bilbyargparser.BilbyArgParser(
        prog=prog,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        add_help=False,
        parents=[base_parser, bilby_pipe_parser]       
    )

    joint_generation_pbilby_parser = _add_slurm_settings_to_parser(joint_generation_pbilby_parser)
    joint_generation_pbilby_parser = _add_misc_settings_to_parser(joint_generation_pbilby_parser)
    joint_generation_pbilby_parser = _add_dynesty_settings_to_parser(joint_generation_pbilby_parser)
    joint_generation_pbilby_parser = _add_hanabi_settings_to_parser(joint_generation_pbilby_parser)

    purge_empty_argument_group(joint_generation_pbilby_parser)

    return joint_generation_pbilby_parser

def create_joint_analysis_pbilby_parser(prog, prog_version):
    parser = create_joint_generation_pbilby_parser(prog, prog_version)

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

def print_unrecognized_arguments(unknown_args, logger):
    if len(unknown_args) > 0:
        msg = [bilby_pipe.utils.tcolors.WARNING, f"Unrecognized arguments {unknown_args}", bilby_pipe.utils.tcolors.END]
        logger.warning(" ".join(msg))
