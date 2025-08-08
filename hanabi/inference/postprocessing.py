import sys
import os
import numpy as np
import pandas as pd
import copy
import bilby
from bilby.core.result import Result, read_in_result
from bilby.core.result import reweight
from bilby.gw.prior import convert_to_flat_in_component_mass_prior, UniformComovingVolume
from ..lensing.conversion import convert_to_lal_binary_black_hole_parameters_for_lensed_BBH
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import parse_args, convert_prior_string_input
import configargparse
import logging
from .utils import ParameterSuffix, load_run_from_pbilby, load_run_from_bilby
from .utils import setup_logger
from scipy.special import logsumexp
from schwimmbad import MultiPool

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
        "--output-filename",
        type=str,
        help="Save output with this filename",
    )

    generating_derived_parameters_parser = parser.add_argument_group(title="Generating derived parameters")
    generating_derived_parameters_parser.add(
        "--generate-component-mass-parameters",
        action="store_true",
        default=False,
        help="Generate samples of component masses if missing",
    )

    prior_reweighting_parser = parser.add_argument_group(title="Prior reweighting")
    prior_reweighting_parser.add(
        "--flat-in-component-masses",
        action="store_true",
        default=False,
        help=(
            "Reweight the result to follow a flat prior in component masses. "
            "It is recommended to use directly `UniformInComponentsChirpMass' and "
            "`UniformInComponentsMassRatio' in sampling"
        )
    )
    prior_reweighting_parser.add(
        "--uniform-in-comoving-volume",
        action="store_true",
        default=False,
        help="Reweight the result to follow a uniform-in-comoving-volume prior in luminosity distance"
    )

    prior_reweighting_parser.add(
        "--reweight-to-prior",
        type=str,
        help="Reweight the prior according to this prior file",
    )

    reconstruction_parser = parser.add_argument_group(title="Reconstructing samples for marginalized parameter(s)")
    reconstruction_parser.add(
        "--generate-samples-for-marginalized-parameters",
        action="store_true",
        default=False,
        help="Reconstruct the full posterior samples for marginalized parameter(s) such as luminosity distance",
    )
    reconstruction_parser.add(
        "--generate-snrs",
        action="store_true",
        default=False,
        help="Generate matched filter and optimal SNRs",
    )
    reconstruction_parser.add(
        "--generate-sky-frame-parameters",
        action="store_true",
        default=False,
        help="Generate sky frame parameters if missing",
    )
    reconstruction_parser.add(
        "--n-triggers",
        type=int,
        default=1,
        help="Number of triggers analyzed jointly",
    )
    reconstruction_parser.add(
        "--not-from-hanabi",
        action="store_true",
        default=False,
        help="The inference was done using vanilla bilby instead of hanabi.inference",
    )
    reconstruction_parser.add(
        "--trigger-ini-files",
        action="append",
        help=(
            "A list of configuration ini files for each trigger analyzed, "
            "specified either by `trigger-ini-files=[PATH_1, PATH2]` or "
            "as command-line arguments by `--trigger-ini-files PATH1 --trigger-ini-files PATH2`"
        )
    )
    reconstruction_parser.add(
        "--data-dump-files",
        action="append",
        help=(
            "A list of data dump files for each trigger, "
            "specified either by `data-dump-files=[FILE1, FILE2]` or "
            "as command-line arguments by `--data-dump-files FILE1 --data-dump-files FILE2`"
        )
    )
    reconstruction_parser.add(
        "--ncores",
        type=int,
        default=1,
        help="Use multi-processing",
    )

    return parser

def reconstruct_likelihoods(n_triggers, trigger_ini_files, data_dump_files, result_from_pbilby=False):
    single_trigger_likelihoods = []
    
    if result_from_pbilby:
        for i in range(n_triggers):
            l, _, _ = load_run_from_pbilby(data_dump_files[i])
            single_trigger_likelihoods.append(l)
    else:
        for i in range(n_triggers):
            l, _, _ = load_run_from_bilby(data_dump_files[i], trigger_ini_files[i])
            single_trigger_likelihoods.append(l)

    return single_trigger_likelihoods

def generate_component_mass_parameters(result):
    logger = logging.getLogger(__prog__)
    if not all([k in result.posterior.columns for k in ["mass_1", "mass_2"]]):
        # Missing mass_1 mass_2 columns
        converted_posterior, added_keys = convert_to_lal_binary_black_hole_parameters_for_lensed_BBH(result.posterior)
        result.posterior = converted_posterior
        logger.info("Adding {} samples".format(", ".join(added_keys)))
    return result

def generate_sky_frame_parameters(result, likelihood):
    logger = logging.getLogger(__prog__)
    # NOTE This function is only for single trigger results and edits the result in-place
    bilby.gw.conversion.generate_sky_frame_parameters(
        result.posterior,
        likelihood
    )
    return result

def reweight_to_prior(result, new_priors):
    old_priors = result.priors
    target_priors = old_priors.copy()

    for param in old_priors.keys():
        try:
            target_priors[param] = new_priors[param]
        except:
            continue
    target_priors = bilby.core.prior.PriorDict(target_priors)

    logger = logging.getLogger(__prog__)
    logger.info("Reweighting to the following prior")
    logger.info(target_priors)

    params = list(old_priors.keys())
    ln_weights = target_priors.ln_prob(result.posterior[params], axis=0) - old_priors.ln_prob(result.posterior[params], axis=0)
    result_reweighted = copy.deepcopy(result)
    result_reweighted.priors = target_priors
    result_reweighted.posterior = bilby.result.rejection_sample(result_reweighted.posterior, np.exp(ln_weights))
    result_reweighted.log_evidence = result.log_evidence + logsumexp(ln_weights) - np.log(len(result.posterior))
    result_reweighted.log_bayes_factor = result.log_bayes_factor + logsumexp(ln_weights) - np.log(len(result.posterior))

    return result_reweighted

def reweight_flat_in_component_masses(result):
    logger = logging.getLogger(__prog__)
    logger.info("Reweighting prior to be flat in component masses")
    result = generate_component_mass_parameters(result)
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
    logger = logging.getLogger(__prog__)
    logger.info("Reweighting prior to be uniform in comoving volume")
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

    result_reweighted = reweight_to_prior(result, new_priors)
    return result_reweighted

def generate_snrs_per_sample(sample, likelihood):
    bilby.gw.conversion.compute_snrs(sample, likelihood)
    return sample

def generate_snrs(result, likelihood, ncores=1):
    logger = logging.getLogger(__prog__)
    logger.info("Using {} CPU core(s) for computing SNRs".format(ncores))
    import tqdm
    with MultiPool(ncores) as pool:
        output_samples = pool.starmap(
            generate_snrs_per_sample,
            tqdm.tqdm([[row.to_dict(), likelihood] for _, row in result.posterior.iterrows()])
        )
    result.posterior = pd.DataFrame(output_samples)
    return result

def generate_joint_snrs_per_sample(
    joint_sample,
    single_trigger_likelihoods,
    common_parameters,
    independent_parameters,
    sep_char,
):
    suffix = ParameterSuffix(sep_char)
    pos_in = joint_sample[common_parameters].to_dict()
    for idx, likelihood in enumerate(single_trigger_likelihoods):
        for p in [q for q in independent_parameters if q.endswith(suffix(idx))]:
            # Fill in the appropriate values
            pos_in[p.replace(suffix(idx), "")] = joint_sample[p]
        pos_out = generate_snrs_per_sample(pos_in, likelihood)

        # Add SNRs
        for p in [q for q in pos_out.keys() if "snr" in q]:
            joint_sample[p+suffix(idx)] = pos_out[p]

    return joint_sample.to_dict()

def generate_joint_snrs(
    joint_result,
    single_trigger_likelihoods,
    sep_char="^",
    ncores=1,
):
    common_parameters = [p for p in list(joint_result.posterior.columns) if sep_char not in p]
    independent_parameters = [p for p in list(joint_result.posterior.columns) if sep_char in p]
    
    logger = logging.getLogger(__prog__)
    logger.info("Using {} CPU core(s) for computing SNRs".format(ncores))
    import tqdm
    with MultiPool(ncores) as pool:
        output_samples = pool.starmap(
            generate_joint_snrs_per_sample,
            tqdm.tqdm([[row, single_trigger_likelihoods, common_parameters, independent_parameters, sep_char] for _, row in joint_result.posterior.iterrows()])
        )

    # Edit data frame
    joint_result.posterior = pd.DataFrame(output_samples)
    return joint_result

def generate_posterior_samples_from_marginalized_likelihood(
    result,
    likelihood,
    ncores=1,
):
    # Use the routine in bilby.gw.conversion instead
    pos_out = bilby.gw.conversion.generate_posterior_samples_from_marginalized_likelihood(
        result.posterior,
        likelihood,
        npool=ncores,
    )
    
    # Update posterior
    result.posterior = pos_out
    # Update prior
    _priors = bilby.core.prior.PriorDict(filename=result.meta_data["command_line_args"]["prior_file"])
    if result.meta_data["command_line_args"]["prior_dict"] is not None:
        _priors.update(
            bilby.core.prior.PriorDict(
                dictionary=convert_prior_string_input(result.meta_data["command_line_args"]["prior_dict"])
            )
        )
    if likelihood.distance_marginalization:
        # Update the _luminosity_ distance prior
        result.priors.update({"luminosity_distance": _priors["luminosity_distance"]})
    if likelihood.phase_marginalization:
        # Update the phase prior
        result.priors.update({"phase": _priors["phase"]})
    if likelihood.time_marginalization:
        # Update the time prior
        result.priors.update({"geocent_time": _priors["geocent_time"]})

    return result

def generate_joint_posterior_sample_from_marginalized_likelihood(
    joint_posterior_sample_from_marginalized_likelihood,
    single_trigger_likelihoods,
    common_parameters,
    independent_parameters,
    sep_char,
):
    suffix = ParameterSuffix(sep_char)
    pos_in = joint_posterior_sample_from_marginalized_likelihood[common_parameters].to_dict()
    for idx, likelihood in enumerate(single_trigger_likelihoods):
        for p in [q for q in independent_parameters if q.endswith(suffix(idx))]:
            # Fill in the appropriate values
            pos_in[p.replace(suffix(idx), "")] = joint_posterior_sample_from_marginalized_likelihood[p]
            
        likelihood.parameters.update(pos_in)
        pos_out = likelihood.generate_posterior_sample_from_marginalized_likelihood()
        for p in [q for q in independent_parameters if q.endswith(suffix(idx))]:
            # Replace with the newly generated sample
            pos_in[p] = pos_out[p.replace(suffix(idx), "")]
            del pos_in[p.replace(suffix(idx), "")]
    return pos_in

def generate_joint_posterior_samples_from_marginalized_likelihood(
    joint_result,
    single_trigger_likelihoods,
    sep_char="^",
    ncores=1,
):
    common_parameters = [p for p in list(joint_result.posterior.columns) if sep_char not in p]
    independent_parameters = [p for p in list(joint_result.posterior.columns) if sep_char in p]
    
    logger = logging.getLogger(__prog__)
    logger.info("Using {} CPU core(s) for generating joint posterior samples from marginalized likelihood".format(ncores))
    import tqdm
    with MultiPool(ncores) as pool:
        output_samples = pool.starmap(
            generate_joint_posterior_sample_from_marginalized_likelihood,
            tqdm.tqdm([[row, single_trigger_likelihoods, common_parameters, independent_parameters, sep_char] for _, row in joint_result.posterior.iterrows()])
        )

    # Edit data frame
    joint_result.posterior = pd.DataFrame(output_samples)
    return joint_result

def main():
    args, unknown_args = parse_args(sys.argv[1:], create_parser(__prog__))
    logger = logging.getLogger(__prog__)

    result = read_in_result(args.result)
    label = result.label

    # Test if the result was generated by parallel_bilby
    result_from_pbilby = False
    try:
        if result.meta_data["command_line_args"]["sampler"] == "parallel_bilby":
            result_from_pbilby = True
    except:
        pass

    if args.generate_samples_for_marginalized_parameters:
        single_trigger_likelihoods = reconstruct_likelihoods(
            n_triggers=args.n_triggers,
            trigger_ini_files=args.trigger_ini_files,
            data_dump_files=args.data_dump_files,
            result_from_pbilby=result_from_pbilby,
        )
        
        if not args.not_from_hanabi:
            result = generate_joint_posterior_samples_from_marginalized_likelihood(result, single_trigger_likelihoods, ncores=args.ncores)
        else:
            if args.n_triggers != 1:
                raise ValueError("Does not understand input")
            result = generate_posterior_samples_from_marginalized_likelihood(result, single_trigger_likelihoods[0], ncores=args.ncores)
        label += "_marginalized_parameter_reconstructed"

    if args.generate_sky_frame_parameters:
        single_trigger_likelihoods = reconstruct_likelihoods(
            n_triggers=args.n_triggers,
            trigger_ini_files=args.trigger_ini_files,
            data_dump_files=args.data_dump_files,
            result_from_pbilby=result_from_pbilby,
        )
        if args.not_from_hanabi:
            if args.n_triggers == 1:
                result = generate_sky_frame_parameters(result, single_trigger_likelihoods[0])
            else:
                raise ValueError("Does not understand input")
        else:
            raise NotImplementedError("Joint sky frame parameters are not implemented yet")

    if args.generate_component_mass_parameters:
        # Edit file **in-place**
        result = generate_component_mass_parameters(result)

    if args.generate_snrs:
        single_trigger_likelihoods = reconstruct_likelihoods(
            n_triggers=args.n_triggers,
            trigger_ini_files=args.trigger_ini_files,
            data_dump_files=args.data_dump_files,
            result_from_pbilby=result_from_pbilby,
        )
        
        # Edit file **in-place**
        if not args.not_from_hanabi:
            result = generate_joint_snrs(result, single_trigger_likelihoods, ncores=args.ncores)
        else:
            if args.n_triggers != 1:
                raise ValueError("Does not understand input")
            result = generate_snrs(result, single_trigger_likelihoods[0], ncores=args.ncores)

    if args.reweight_to_prior is not None:
        result = reweight_to_prior(result, bilby.core.prior.PriorDict(filename=args.reweight_to_prior))
        label += "_reweighted"

    if args.flat_in_component_masses:
        result = reweight_flat_in_component_masses(result)
        label += "_reweighted"

    if args.uniform_in_comoving_volume:
        result = reweight_uniform_in_comoving_volume(result)
        label += "_reweighted"

    # Save to file
    logger.info("Done. Saving to file")
    result.label = label
    result.save_to_file(
        outdir=".",
        filename=args.output_filename,
        extension=os.path.splitext(args.output_filename)[1].split('.')[1]
    )
