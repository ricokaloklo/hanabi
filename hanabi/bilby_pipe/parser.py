import bilby
import bilby_pipe
import bilby_pipe.bilbyargparser
from bilby_pipe.parser import StoreBoolean
from bilby_pipe.utils import nonestr, nonefloat, noneint
import configargparse
import logging


def create_joint_parser(prog, prog_version):
    """
    Create a parser to read the joint analysis config ini
    """
    
    parser = bilby_pipe.bilbyargparser.BilbyArgParser(
        prog=prog,
        usage=None,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Setting up the parser
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file for the joint analysis")
    parser.add("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={prog_version}\nbilby={bilby.__version__}\nbilby_pipe={bilby_pipe.__version__}"
    )

    # Copied from bilby_pipe
    submission_parser = parser.add_argument_group(
        title="Job submission arguments",
        description="How the jobs should be formatted, e.g., which job scheduler to use.",
    )
    submission_parser.add(
        "--accounting",
        type=nonestr,
        help="Accounting group to use (see, https://accounting.ligo.org/user)",
    )
    submission_parser.add("--label", type=str, default="label", help="Output label")
    submission_parser.add(
        "--local",
        action="store_true",
        help="Run the job locally, i.e., not through a batch submission",
    )
    submission_parser.add(
        "--local-generation",
        action="store_true",
        help=(
            "Run the data generation job locally. This may be useful for "
            "running on a cluster where the compute nodes do not have "
            "internet access. For HTCondor, this is done using the local "
            "universe, for slurm, the jobs will be run at run-time"
        ),
    )
    submission_parser.add(
        "--local-plot", action="store_true", help="Run the plot job locally"
    )

    submission_parser.add("--outdir", type=str, default=".", help="Output directory")
    submission_parser.add(
        "--periodic-restart-time",
        default=28800,
        type=int,
        help=(
            "Time after which the job will self-evict when scheduler=condor."
            " After this, condor will restart the job. Default is 28800."
            " This is used to decrease the chance of HTCondor hard evictions"
        ),
    )
    submission_parser.add(
        "--request-memory",
        type=float,
        default=4.0,
        help="Memory allocation request (GB), defaults is 4GB",
    )
    submission_parser.add(
        "--request-memory-generation",
        type=nonefloat,
        default=None,
        help="Memory allocation request (GB) for data generation step",
    )
    submission_parser.add(
        "--request-cpus",
        type=int,
        default=1,
        help="Use multi-processing (for available samplers: dynesty, ptemcee, cpnest)",
    )
    submission_parser.add(
        "--singularity-image",
        type=nonestr,
        default=None,
        help="Singularity image to use",
    )
    submission_parser.add(
        "--scheduler",
        type=str,
        default="condor",
        help="Format submission script for specified scheduler. Currently implemented: SLURM",
    )
    submission_parser.add(
        "--scheduler-args",
        type=nonestr,
        default=None,
        help="Space-separated #SBATCH command line args to pass to slurm (slurm only)",
    )
    submission_parser.add(
        "--scheduler-module",
        type=nonestr,
        action="append",
        default=None,
        help="Space-separated list of modules to load at runtime (slurm only)",
    )
    submission_parser.add(
        "--scheduler-env",
        type=nonestr,
        default=None,
        help="Python environment to activate (slurm only)",
    )
    submission_parser.add(
        "--scheduler-analysis-time", type=nonestr, default="7-00:00:00", help=""
    )
    submission_parser.add(
        "--submit",
        action="store_true",
        help="Attempt to submit the job after the build",
    )
    submission_parser.add(
        "--condor-job-priority",
        type=int,
        default=0,
        help=(
            "Job priorities allow a user to sort their HTCondor jobs to determine "
            "which are tried to be run first. "
            "A job priority can be any integer: larger values denote better priority. "
            "By default HTCondor job priority=0. "
        ),
    )
    submission_parser.add(
        "--transfer-files",
        action=StoreBoolean,
        default=True,
        help=(
            "If true, use HTCondor file transfer mechanism, default is True"
            "for non-condor schedulers, this option is ignored"
        ),
    )
    submission_parser.add(
        "--log-directory",
        type=nonestr,
        default=None,
        help="If given, an alternative path for the log output",
    )
    submission_parser.add(
        "--online-pe", action="store_true", help="Flag for online PE settings"
    )
    submission_parser.add(
        "--osg",
        action="store_true",
        default=False,
        help="If true, format condor submission for running on OSG, default is False",
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
            "A list of common parameters that are forced to take the same value across different triggers, "
            "specified either by `common-parameters=[chirp_mass, mass_ratio]` or "
            "as command-line arguments by `--common-parameters chirp_mass --common-parameters mass_ratio`"
        )
    )
    lensing_prior_parser.add(
        "--lensing-prior-dict",
        type=nonestr,
        default=None,
        help=(
            "A dictionary of prior for lensing magnification and image type. If the magnification of the first trigger "
            "is set to 1, then the magnification is interpreted as the relative magnification. "
            "Otherwise, the magnification is interpreted as the absolute magnification. "
            "Specified by lensing-prior-dict={magnification^1 = 1, magnification^2 = PowerLaw(...)} for relative magnification, or "
            "lensing-prior-dict={magnification^1 = PowerLaw(...), magnification^2 = PowerLaw(...)} for absolute magnification"
        )
    )

    return parser

def print_unrecognized_arguments(unknown_args, logger):
    if len(unknown_args) > 0:
        msg = [bilby_pipe.utils.tcolors.WARNING, f"Unrecognized arguments {unknown_args}", bilby_pipe.utils.tcolors.END]
        logger.warning(" ".join(msg))
