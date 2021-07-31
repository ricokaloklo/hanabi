import numpy as np
import bilby
import h5py
from ..inference.utils import setup_logger

def write_to_hdf5(filename, dataset_dict, attrs_dict):
    with h5py.File(filename, "w") as f:
        # Write attributes
        for k,v in attrs_dict.items():
            f.attrs[k] = v

        # Write dataset
        for k,v in dataset_dict.items():
            # v itself is also a dict of columns
            dataset_dt = np.dtype({"names": list(v.keys()), "formats": [d.dtype for d in v.values()]})
            dataset = np.rec.fromarrays([d for d in v.values()], dtype=dataset_dt)
            f.create_dataset(k, data=dataset, chunks=True, compression="gzip", compression_opts=9)

def enforce_mass_ordering(m1, m2):
    for idx, (m1_unordered, m2_unordered) in enumerate(zip(m1, m2)):
        if m1_unordered < m2_unordered:
            tmp = m1_unordered
            m1[idx] = m2_unordered
            m2[idx] = tmp
    return m1, m2

# This is a stripped-down version of bilby.core.result.get_weights_for_reweighting
# Currently the function in bilby v1.0.2 is unusable if rejection sampling was used before
def get_ln_weights_for_reweighting(result, old_priors, new_priors, parameters):
    samples = {key: result.posterior[key].to_numpy() for key in parameters}
    old_log_prior_array = old_priors.ln_prob(samples, axis=0)
    new_log_prior_array = new_priors.ln_prob(samples, axis=0)

    ln_weights = new_log_prior_array - old_log_prior_array
    return ln_weights

# Initialize a logger for hanabi_hierarchical_analysis
setup_logger("hanabi_hierarchical_analysis")
# Initialize a logger for selection_function.py
setup_logger("selection_function.py")