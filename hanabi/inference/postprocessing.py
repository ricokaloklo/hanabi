import sys
from bilby.core.result import Result
from bilby.core.result import reweight as reweigh
from bilby.gw.prior import convert_to_flat_in_component_mass_prior, UniformComovingVolume
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
    parser.add("result", type=str, help="The result file")
    parser.add(
        "--flat-in-component-masses",
        action="store_true",
        default=False,
        help="Reweigh the result to follow a flat prior in component masses"
    )
    parser.add(
        "--uniform-in-comoving-volume",
        action="store_true",
        default=False,
        help="Reweigh the result to follow a uniform-in-comoving-volume prior in luminosity distance"
    )

    return parser

def reweigh_flat_in_component_masses(result):
    result_reweighed = convert_to_flat_in_component_mass_prior(result)
    return result_reweighed

def reweigh_uniform_in_comoving_volume(result):
    # Read in the old priors
    old_priors = result.priors.copy()

    # Construct new priors
    new_priors = result.priors.copy()
    for name in new_priors.keys():
        if "luminosity_distance" in name:
            new_priors[name] = UniformComovingVolume(
                name=old_priors[name].name,
                minimum=old_priors[name].minimum,
                maximum=old_priors[name].maximum,
                unit=old_priors[name].unit,
                latex_label=old_priors[name].latex_label
            )

    result_reweighed = reweigh(result, old_prior=old_priors, new_prior=new_priors)

    return result_reweighed

def main():
    args, unknown_args = parse_args(sys.argv[1:], create_parser(__prog__))

    result = Result.from_json(args.result)
    label = result.label

    if args.flat_in_component_masses:
        result = reweigh_flat_in_component_masses(result)

    if args.uniform_in_comoving_volume:
        result = reweigh_uniform_in_comoving_volume(result)

    # Save to file
    result.label = label + "_reweighed"
    result.save_to_file()

