import bilby_pipe
import logging

# Useful routine from parallel_bilby
def remove_argument_from_parser(parser, arg, prog_name):
    logger = logging.getLogger(prog_name)

    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            try:
                parser._handle_conflict_resolve(None, [("--" + arg, action)])
            except ValueError as e:
                logger.warning("Error removing {}: {}".format(arg, e))
    logger.debug(
        "Request to remove arg {} from bilby_pipe args, but arg not found".format(arg)
    )


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

# Initialize a logger for hanabi_joint_pipe
setup_logger("hanabi_joint_pipe")
# Initialize a logger for hanabi_joint_analysis
setup_logger("hanabi_joint_analysis")
# Initialize a logger for hanabi_joint_analysis_pbilby
setup_logger("hanabi_joint_analysis_pbilby")