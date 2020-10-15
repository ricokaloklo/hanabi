import sys
from bilby.core.result import Result
from bilby.core.result import reweight
from bilby.gw.prior import convert_to_flat_in_component_mass_prior, UniformComovingVolume, PowerLaw
from ..lensing.prior import RelativeMagnificationPoorMan
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
        help="Reweight the result to follow a flat prior in component masses"
    )
    parser.add(
        "--uniform-in-comoving-volume",
        action="store_true",
        default=False,
        help="Reweight the result to follow a uniform-in-comoving-volume prior in luminosity distance"
    )
    parser.add(
        "--convert-to-relative-magnification",
        action="store_true",
        default=False,
        help=(
            "Reweight the result with luminosity distances being sampled independently"
            "to relative magnification following the poor man's prior."
            "Note that the prior used during sampling must be PowerLaw with exponent of 2 (d_L^2)"
        )
    )

    return parser

def reweight_flat_in_component_masses(result):
    result_reweighted = convert_to_flat_in_component_mass_prior(result)
    return result_reweighted

def reweight_uniform_in_comoving_volume(result):
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

    result_reweighted = reweight(result, old_prior=old_priors, new_prior=new_priors)

    return result_reweighted

def main():
    args, unknown_args = parse_args(sys.argv[1:], create_parser(__prog__))

    result = Result.from_json(args.result)
    label = result.label

    if args.flat_in_component_masses:
        result = reweight_flat_in_component_masses(result)

    if args.uniform_in_comoving_volume:
        result = reweight_uniform_in_comoving_volume(result)

    # Save to file
    result.label = label + "_reweighted"
    result.save_to_file()

