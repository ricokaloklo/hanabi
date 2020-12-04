import numpy as np
import bilby

# This is a stripped-down version of bilby.core.result.get_weights_for_reweighting
# Currently the function in bilby v1.0.2 is unusable if rejection sampling was used before
def get_ln_weights_for_reweighting(result, old_priors, new_priors, parameters):
    n_posteriors = len(result.posterior)
    old_log_prior_array = np.zeros(n_posteriors)
    new_log_prior_array = np.zeros(n_posteriors)

    par_samples = {key: result.posterior[key] for key in parameters}
    old_log_prior_array = old_priors.ln_prob(par_samples)
    new_log_prior_array = new_priors.ln_prob(par_samples)

    ln_weights = new_log_prior_array - old_log_prior_array
    return ln_weights