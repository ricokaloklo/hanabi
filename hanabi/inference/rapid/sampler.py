import numpy as np
import pandas as pd
import itertools
from scipy.special import logsumexp

def sample_time_dist_marginalized(
        theta,
        trigger_ids,
        suffix,
        likelihood_parameter_keys,
        independent_parameters,
        lensing_prior_dict,
        likelihood_base,
        single_trigger_likelihoods_with_cache,
        waveform_cache=True
):
    theta_dict = {p: theta[p] for p in likelihood_parameter_keys}

    if waveform_cache:
        likelihood_base.parameters.update(theta_dict)
        likelihood_base.log_likelihood_ratio()

    log_evidence = 0.
    posterior_to_add = {}

    for trigger_idx in trigger_ids[1:]:
        conditioned_likelihood = single_trigger_likelihoods_with_cache[trigger_idx]
        conditioned_likelihood.parameters.update(theta_dict)

        if waveform_cache:
            # Assign waveform cache (so that we do not need to re-evaluate waveform)
            conditioned_likelihood.initialize_cache(likelihood_base.waveform_generator._cache, theta_dict)

        conditioned_likelihood.parameters.update({
                "geocent_time": float(conditioned_likelihood.interferometers.start_time)
        })
        
        # For each image type, compute the logL
        image_types = [1.0, 2.0, 3.0]
        log_priors = []
        log_Ls = []            

        for image_type in image_types:
            log_priors.append(
                lensing_prior_dict["image_type"+suffix(trigger_idx)].ln_prob(image_type)
            )
            conditioned_likelihood.parameters.update({'image_type': image_type})
            log_Ls.append(conditioned_likelihood.log_likelihood())

        # All parameters are marginalized over except for image type
        log_priors = np.array(log_priors)
        log_Ls = np.array(log_Ls)

        log_evidence += logsumexp(log_Ls, b=np.exp(log_priors))

        # Draw one posterior sample for each image type, weighted by log posterior
        drawn_image_type = np.random.choice(image_types, p=np.exp(log_Ls + log_priors - log_evidence))
        conditioned_likelihood.parameters.update({'image_type': drawn_image_type})
        drawn_sample = conditioned_likelihood.generate_posterior_sample_from_marginalized_likelihood()

        # Rename independent parameters
        for k in independent_parameters:
            drawn_sample[k+suffix(trigger_idx)] = drawn_sample.pop(k)
        posterior_to_add.update(drawn_sample)

    for k in independent_parameters:
        posterior_to_add[k+suffix(trigger_ids[0])] = theta_dict[k] 

    return log_evidence, posterior_to_add

def generate_independent_parameters_per_image_type(
        likelihood,
        theta,
        ref_distance,
        image_type
):
    likelihood.parameters.update(theta)
    likelihood.parameters.update({
        'image_type': image_type,
        'luminosity_distance': ref_distance,
        'geocent_time': float(likelihood.interferometers.start_time),
    })
    return likelihood.generate_posterior_sample_from_marginalized_likelihood()

def generate_all_parameters(
        theta,
        trigger_ids,
        suffix,
        independent_parameters,
        lensing_prior_dict,
        single_trigger_priors,
        single_trigger_likelihoods,
        single_trigger_likelihoods_with_cache,
):
    joint_log_priors = []
    joint_log_Ls = []
    # First generate samples for time and distance as if the signal is of type I
    posterior = dict(theta)
    for trigger_idx in trigger_ids:
        parameters = generate_independent_parameters_per_image_type(
            likelihood=single_trigger_likelihoods_with_cache[trigger_idx],
            theta=theta,
            ref_distance=single_trigger_priors[trigger_idx]["luminosity_distance"].rescale(0.5),
            image_type=1.0,
        )
        for p in independent_parameters:
            posterior[p+suffix(trigger_idx)] = parameters[p]

    possible_image_types = [1.0, 2.0, 3.0]
    image_type_combos = list(tuple(itertools.product(possible_image_types, repeat=len(trigger_ids))))
    # Then iterate over all possible combinations
    for image_type_combo in image_type_combos:
        log_prior = 0.
        log_L = 0.
        for trigger_idx in trigger_ids:
            single_trigger_likelihoods[trigger_idx].parameters.update(theta)
            single_trigger_likelihoods[trigger_idx].parameters.update(
                {
                    "image_type": image_type_combo[trigger_idx],
                    "geocent_time": posterior["geocent_time"+suffix(trigger_idx)],
                    "luminosity_distance": posterior["luminosity_distance"+suffix(trigger_idx)],
                }
            )
            log_L += single_trigger_likelihoods[trigger_idx].log_likelihood()
            log_prior += lensing_prior_dict["image_type"+suffix(trigger_idx)].ln_prob(image_type_combo[trigger_idx])
        joint_log_priors.append(log_prior)
        joint_log_Ls.append(log_L)

    joint_log_priors = np.array(joint_log_priors)
    joint_log_Ls = np.array(joint_log_Ls)
    log_norm = logsumexp(joint_log_Ls, b=np.exp(joint_log_priors))

    drawn_image_types = np.random.choice(np.arange(len(image_type_combos)), p=np.exp(joint_log_Ls + joint_log_priors - log_norm))
    posterior.update(
        {"image_type"+suffix(trigger_idx): image_type_combos[drawn_image_types][trigger_idx] for trigger_idx in trigger_ids}
    )
    posterior.update(
        {"log_likelihood": joint_log_Ls[drawn_image_types]}
    )
    return posterior

def lnpriorfn(x, bilby_prior, joint_search_parameter_keys):
    theta_dict = {k: x[idx] for idx, k in enumerate(joint_search_parameter_keys)}
    # NOTE The image_type parameters are *discrete*, need to apply transformation
    for p in theta_dict.keys():
        if p.startswith("image_type"):
            theta_dict[p] = round(theta_dict[p])
            
    log_prior = bilby_prior.ln_prob(theta_dict)
    return log_prior

def lnlikefn(x, bilby_likelihood, joint_search_parameter_keys):
    theta_dict = {k: x[idx] for idx, k in enumerate(joint_search_parameter_keys)}
    # NOTE The image_type parameters are *discrete*, need to apply transformation
    for p in theta_dict.keys():
        if p.startswith("image_type"):
            theta_dict[p] = round(theta_dict[p])

    bilby_likelihood.parameters.update(theta_dict)
    return bilby_likelihood.log_likelihood()

def lnpostfn(x, bilby_likelihood, bilby_prior, joint_search_parameter_keys):
    log_prior = lnpriorfn(x, bilby_prior, joint_search_parameter_keys)
    if not np.isfinite(log_prior):
        return -np.inf
    else:
        log_like = lnlikefn(x, bilby_likelihood, joint_search_parameter_keys)
        if not np.isfinite(log_like):
            return -np.inf
        else:
            return log_like + log_prior