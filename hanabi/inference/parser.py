import bilby
import bilby_pipe
import bilby_pipe.bilbyargparser
from bilby_pipe.parser import StoreBoolean
from bilby_pipe.utils import nonestr, nonefloat, noneint
import configargparse
import logging


# Useful function copied from parallel_bilby
def remove_argument_from_parser(parser, arg, prog):
    logger = logging.getLogger(prog)

    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            try:
                parser._handle_conflict_resolve(None, [("--" + arg, action)])
            except ValueError as e:
                logger.warning("Error removing {}: {}".format(arg, e))
    logger.debug(
        "Request to remove arg {} from bilby_pipe args, but arg not found".format(arg)
    )


def create_joint_main_parser(prog, prog_version):
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
    # Arguments for sampler
    sampler_parser = parser.add_argument_group(title="Sampler arguments")
    sampler_parser.add("--sampler", type=str, default="dynesty", help="Sampler to use")
    sampler_parser.add(
        "--sampling-seed", default=None, type=noneint, help="Random sampling seed"
    )
    sampler_parser.add(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of identical parallel jobs to run per event",
    )
    sampler_parser.add(
        "--sampler-kwargs",
        type=str,
        default="Default",
        help=(
            "Dictionary of sampler-kwargs to pass in, e.g., {nlive: 1000} OR "
            "pass pre-defined set of sampler-kwargs {Default, FastTest}"
        ),
    )

    # Arguments for job submission
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
        default=8.0,
        help="Memory allocation request (GB), defaults is 8GB",
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
        help="Format submission script for specified scheduler. Currently implemented: condor, slurm",
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
        default=False,
        help=(
            "If true, use HTCondor file transfer mechanism, default is False"
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

    # Arguments for output
    output_parser = parser.add_argument_group(
        title="Output arguments", description="What kind of output/summary to generate."
    )
    output_parser.add(
        "--create-plots",
        action="store_true",
        help="Create diagnostic and posterior plots",
    )
    output_parser.add_argument(
        "--plot-calibration",
        action="store_true",
        help="Create calibration posterior plot",
    )
    output_parser.add_argument(
        "--plot-corner",
        action="store_true",
        help="Create intrinsic and extrinsic posterior corner plots",
    )
    output_parser.add_argument(
        "--plot-marginal",
        action="store_true",
        help="Create 1-d marginal posterior plots",
    )
    output_parser.add_argument(
        "--plot-skymap", action="store_true", help="Create posterior skymap"
    )
    output_parser.add_argument(
        "--plot-waveform", action="store_true", help="Create waveform posterior plot"
    )
    output_parser.add_argument(
        "--plot-format",
        default="png",
        help="Format for making bilby_pipe plots, can be [png, pdf, html]. "
        "If specified format is not supported, will default to png.",
    )

    output_parser.add(
        "--create-summary", action="store_true", help="Create a PESummary page"
    )
    output_parser.add("--email", type=nonestr, help="Email for notifications")
    output_parser.add(
        "--notification",
        type=nonestr,
        default="Never",
        help=(
            "Notification setting for HTCondor jobs. "
            "One of 'Always','Complete','Error','Never'. "
            "If defined by 'Always', "
            "the owner will be notified whenever the job "
            "produces a checkpoint, as well as when the job completes. "
            "If defined by 'Complete', "
            "the owner will be notified when the job terminates. "
            "If defined by 'Error', "
            "the owner will only be notified if the job terminates abnormally, "
            "or if the job is placed on hold because of a failure, "
            "and not by user request. "
            "If defined by 'Never' (the default), "
            "the owner will not receive e-mail, regardless to what happens to the job. "
            "Note, an `email` arg is also required for notifications to be emailed. "
        ),
    )
    output_parser.add(
        "--existing-dir",
        type=nonestr,
        default=None,
        help=(
            "If given, add results to an directory with an an existing"
            " summary.html file"
        ),
    )
    output_parser.add(
        "--webdir",
        type=nonestr,
        default=None,
        help=(
            "Directory to store summary pages. If not given, defaults to "
            "outdir/results_page"
        ),
    )
    output_parser.add(
        "--summarypages-arguments",
        type=nonestr,
        default=None,
        help="Arguments (in the form of a dictionary) to pass to the summarypages executable",
    )

    # Arguments for post-processing
    postprocessing_parser = parser.add_argument_group(
        title="Post processing arguments",
        description="What post-processing to perform.",
    )
    postprocessing_parser.add(
        "--postprocessing-executable",
        type=nonestr,
        default=None,
        help=(
            "An executable name for postprocessing. A single postprocessing "
            " job is run as a child of all analysis jobs"
        ),
    )
    postprocessing_parser.add(
        "--postprocessing-arguments",
        type=nonestr,
        default=None,
        help="Arguments to pass to the postprocessing executable",
    )
    postprocessing_parser.add(
        "--single-postprocessing-executable",
        type=nonestr,
        default="hanabi_postprocess_result",
        help=(
            "An executable name for postprocessing. A single postprocessing "
            "job is run as a child for each analysis jobs: note the "
            "difference with respect postprocessing-executable"
        ),
    )
    postprocessing_parser.add(
        "--single-postprocessing-arguments",
        type=nonestr,
        default="--flat-in-component-masses --uniform-in-comoving-volume $RESULT",
        help=(
            "Arguments to pass to the single postprocessing executable. The "
            "str '$RESULT' will be replaced by the path to the individual "
            "result file"
        ),
    )

    # Make hanabi_joint_analysis to retry for the data generation jobs to complete
    submission_parser.add(
        "--retry-for-data-generation",
        type=int,
        default=0,
        help=(
            "Retry an analysis node every N minute to wait for the data generation jobs"
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
            "A dictionary of prior for lensing magnification and image type."
            "Specified by lensing-prior-dict={relative-magnification^(1) = 1, relative-magnification^(2) = PowerLaw(...)} for relative magnification, or "
            "lensing-prior-dict={absolute-magnification^(1) = PowerLaw(...), absolute-magnification^(2) = PowerLaw(...)} for absolute magnification"
        )
    )

    # Lensed waveform arguments
    lensed_waveform_parser = parser.add_argument_group(
        title="Waveform arguments", description="Setting for the waveform generator for lensed signals"
    )

    lensed_waveform_parser.add(
        "--lensed-waveform-model",
        default="strongly_lensed_BBH_waveform",
        type=str,
        help=(
            "Name of the lensed waveform model. Can be one of"
            "[strongly_lensed_BBH_waveform] or any python  path to a bilby "
            " source function the users installation, e.g. examp.source.bbh"
        ),
    )

    return parser

def print_unrecognized_arguments(unknown_args, logger):
    if len(unknown_args) > 0:
        msg = [bilby_pipe.utils.tcolors.WARNING, f"Unrecognized arguments {unknown_args}", bilby_pipe.utils.tcolors.END]
        logger.warning(" ".join(msg))
