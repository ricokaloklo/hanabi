import numpy as np
import bilby

# This is a stripped-down version of bilby.core.result.get_weights_for_reweighting
# Currently the function in bilby v1.0.2 is unusable if rejection sampling was used before
def get_weights_for_reweighting(result, old_priors, new_priors, parameters):
    n_posteriors = len(result.posterior)
    old_log_prior_array = np.zeros(n_posteriors)
    new_log_prior_array = np.zeros(n_posteriors)

    for idx, (_, sample) in enumerate(result.posterior.iterrows()):
        par_sample = {key: sample[key] for key in parameters}
        old_log_prior_array[idx] = old_priors.ln_prob(par_sample)
        new_log_prior_array[idx] = new_priors.ln_prob(par_sample)

    ln_weights = new_log_prior_array - old_log_prior_array
    return np.exp(ln_weights)