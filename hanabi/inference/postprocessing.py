import sys
import numpy as np
from bilby.core.result import Result
from bilby.core.result import reweight
from bilby.gw.prior import convert_to_flat_in_component_mass_prior, UniformComovingVolume
from ..lensing.prior import RelativeMagnificationPoorMan
from ..lensing.conversion import convert_to_lal_binary_black_hole_parameters_for_lensed_BBH
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import parse_args
import configargparse
import logging
from .utils import setup_logger
from scipy.special import logsumexp

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

    return parser

def reweight_flat_in_component_masses(result):
    logger = logging.getLogger(__prog__)
    if not all([k in result.posterior.columns for k in ["mass_1", "mass_2"]]):
        # Missing mass_1 mass_2 columns
        converted_posterior, added_keys = convert_to_lal_binary_black_hole_parameters_for_lensed_BBH(result.posterior)
        result.posterior = converted_posterior
        logger.info("Adding {} samples".format(", ".join(added_keys)))
    result_reweighted = convert_to_flat_in_component_mass_prior(result)
    # NOTE Here we reweight the **evidence** as well
    weights = np.array(result.get_weights_by_new_prior(result.priors, result_reweighted.priors,
                                                       prior_names=['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']))
    jacobian = result.posterior["mass_1"] ** 2 / result.posterior["chirp_mass"]
    ln_weights = np.log(jacobian * weights)
    result_reweighted.log_evidence = result.log_evidence + logsumexp(ln_weights) - np.log(len(result.posterior))
    result_reweighted.log_bayes_factor = result.log_bayes_factor + logsumexp(ln_weights) - np.log(len(result.posterior))
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
    result.save_to_file(outdir=".")

