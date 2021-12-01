import numpy as np
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