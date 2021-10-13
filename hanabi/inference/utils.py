import bilby
import bilby_pipe
import logging
import os
from pathlib import Path
import inspect
import pickle
from importlib import import_module

from bilby.gw.likelihood import GravitationalWaveTransient
from .._version import __version__

class ParameterSuffix(object):
    def __init__(self, sep_char="^"):
        self.sep_char = sep_char

    def __call__(self, trigger_idx):
        return "{}({})".format(self.sep_char, trigger_idx + 1)

def turn_off_forbidden_option(input, forbidden_option, prog):
    # NOTE Only support boolean option
    if getattr(input, forbidden_option, False):
        logger = logging.getLogger(prog)
        logger.info(f"Turning off {forbidden_option}")
        setattr(input, forbidden_option, False)


def write_complete_config_file(parser, args, inputs, prog):
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
    parser.write_to_file(
        filename=inputs.complete_ini_file,
        args=args,
        overwrite=False,
        include_description=False,
    )

    logger = logging.getLogger(prog)
    logger.info(f"Complete ini written: {inputs.complete_ini_file}")

# The following code is modified from bilby_pipe.utils
def setup_logger(prog_name, outdir=None, label=None, log_level="INFO"):
    """Setup logging output: call at the start of the script to use

    Parameters
    ----------
    prog_name: str
        Name of the program
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError(f"log_level {log_level} not understood")
    else:
        level = int(log_level)

    logger = logging.getLogger(prog_name)
    logger.propagate = False
    logger.setLevel(level)

    streams = [isinstance(h, logging.StreamHandler) for h in logger.handlers]
    if len(streams) == 0 or not all(streams):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
            )
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([isinstance(h, logging.FileHandler) for h in logger.handlers]) is False:
        if label:
            if outdir:
                bilby_pipe.utils.check_directory_exists_and_if_not_mkdir(outdir)
            else:
                outdir = "."
            log_file = f"{outdir}/{label}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
                )
            )

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

def get_version_information():
    version_file = Path(__file__).parents[1] / ".version"
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except FileNotFoundError:
        print("No version information file '.version' found")
        return __version__

def setup_likelihood_from_pbilby(interferometers, waveform_generator, priors, args):
    """Takes the kwargs and sets up and returns  either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        The waveform generation
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        The likelihood (either GravitationalWaveTransient or ROQGravitationalWaveTransient)

    """

    likelihood_kwargs = dict(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=args.phase_marginalization,
        distance_marginalization=args.distance_marginalization,
        distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
        time_marginalization=args.time_marginalization,
        reference_frame=args.reference_frame,
        time_reference=args.time_reference,
    )

    if args.likelihood_type == "GravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.GravitationalWaveTransient
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient

        if args.time_marginalization:
                """Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"""
                pass

        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))
    elif "." in args.likelihood_type:
        split_path = args.likelihood_type.split(".")
        module = ".".join(split_path[:-1])
        likelihood_class = split_path[-1]
        Likelihood = getattr(import_module(module), likelihood_class)
        likelihood_kwargs.update(args.extra_likelihood_kwargs)
        if "roq" in args.likelihood_type.lower():
            likelihood_kwargs.pop("time_marginalization", None)
            likelihood_kwargs.pop("jitter_time", None)
            likelihood_kwargs.update(args.roq_likelihood_kwargs)
    else:
        raise ValueError("Unknown Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood

def load_run_from_pbilby(data_dump_file, result_file=None, **kwargs):
    if result_file is not None:
        result = bilby.result.read_in_result(result_file)
    else:
        result = None

    with open(data_dump_file, "rb") as f:
        data_dump = pickle.load(f)

    ifo_list = data_dump["ifo_list"]
    waveform_generator = data_dump["waveform_generator"]
    waveform_generator.start_time = ifo_list[0].time_array[0]
    args = data_dump["args"]

    if result is not None:
        priors = result.priors
    else:
        priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])

    # Override args if given in kwargs
    for k, v in kwargs.items():
        setattr(args, k, v)

    likelihood = setup_likelihood_from_pbilby(
        interferometers=ifo_list,
        waveform_generator=waveform_generator,
        priors=priors,
        args=args,
    )

    return likelihood, priors, result

def load_run_from_bilby(data_dump_file, trigger_ini_file, result_file=None, **kwargs):
    if result_file is not None:
        result = bilby.result.read_in_result(result_file)
    else:
        result = None
    args, unknown_args = bilby_pipe.utils.parse_args([trigger_ini_file, "--data-dump-file", data_dump_file], bilby_pipe.data_analysis.create_analysis_parser())

    # Override args if given in kwargs
    for k, v in kwargs.items():
        setattr(args, k, v)

    from hanabi.inference.joint_analysis import SingleTriggerDataAnalysisInput
    single_trigger_analysis = SingleTriggerDataAnalysisInput(args, unknown_args)
    likelihood, priors = single_trigger_analysis.get_likelihood_and_priors()

    return likelihood, priors, result

# Initialize a logger for hanabi_joint_pipe
setup_logger("hanabi_joint_pipe")
# Initialize a logger for hanabi_joint_analysis
setup_logger("hanabi_joint_analysis")
# Initialize a logger for hanabi_joint_generation_pbilby
setup_logger("hanabi_joint_generation_pbilby")
# Initialize a logger for hanabi_joint_analysis_pbilby
setup_logger("hanabi_joint_analysis_pbilby")
# Initialize a logger for hanabi_postprocess_result
setup_logger("hanabi_postprocess_result")