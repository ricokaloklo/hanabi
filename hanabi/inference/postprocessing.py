import sys
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import parse_args
import configargparse

__prog__ = "hanabi_postprocess_result"

def create_parser(prog):
    parser = BilbyArgParser(
        prog=prog,
        usage=None,
        ignore_unknown_config_file_keys=False,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add("--result", type=str, required=True, help="The result file")
    parser.add(
        "--flat-in-component-masses",
        action="store_true",
        default=True,
        help="Reweigh the result to follow a flat prior in component masses"
    )
    parser.add(
        "--uniform-in-comoving-volume",
        action="store_true",
        default=True,
        help="Reweigh the result to follow a uniform-in-comoving-volume prior in luminosity distance"
    )

    return parser

def main():
    args, unknown_args = parse_args(sys.argv[1:], create_parser(__prog__))

