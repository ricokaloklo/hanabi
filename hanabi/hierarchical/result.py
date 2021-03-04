import numpy as np
import pandas as pd
import h5py
import os
import bilby
from bilby.core.result import Result
from .reweight_with_population_model import ReweightWithPopulationModel

def compute_source_parameters(joint_samples):
    # Adding samples in-place
    # Masses
    joint_samples["mass_1_source"] = joint_samples["mass_1"]/(1. + joint_samples["redshift"])
    joint_samples["mass_2_source"] = joint_samples["mass_2"]/(1. + joint_samples["redshift"])
    joint_samples["total_mass_source"] = joint_samples["mass_1_source"] + joint_samples["mass_2_source"]

    # Absolute and relative magnification
    dL_names = [name for name in joint_samples.keys() if "luminosity_distance" in name]
    for p in dL_names:
        joint_samples[p.replace("luminosity_distance", "magnification")] = (bilby.gw.conversion.redshift_to_luminosity_distance(joint_samples["redshift"])/joint_samples[p])**2
        joint_samples[p.replace("luminosity_distance", "relative_magnification")] = joint_samples[p.replace("luminosity_distance", "magnification")]/joint_samples[dL_names[0].replace("luminosity_distance", "magnification")]

def compute_log_coherence_ratio(joint_result, *single_result):
    joint_log_evidence = joint_result.log_evidence
    if isinstance(list(single_result)[0], ReweightWithPopulationModel):
        # Need to compute the reweighted log evidence
        return joint_log_evidence - np.sum([r.reweight_ln_evidence() for r in list(single_result)])
    else:
        return joint_log_evidence - np.sum([r.log_evidence for r in list(single_result)])

def compute_log_Bayes_factor(selection_function_hdf5_file, joint_result, *single_result):
    with h5py.File(selection_function_hdf5_file, "r") as f:
        alpha = f.attrs["alpha"]
        beta = f.attrs["beta"]
    
    log_alphaN_over_beta = np.log(alpha**len(list(single_result))/beta)
    log_coherence_ratio = compute_log_coherence_ratio(joint_result, *single_result)

    return log_alphaN_over_beta + log_coherence_ratio

def save_hierarchical_analysis_result(log_Bayes_factor, log_coherence_ratio, joint_samples, label, outdir="."):
    output_filename = os.path.join(os.path.abspath(outdir), "{}.h5".format(label))
    f = h5py.File(output_filename, "w")
    output_samples = joint_samples.drop(columns=['log_prior']).to_dict(orient='list')
    output_samples_keys = list(output_samples.keys())
    output_samples_dt = np.dtype({"names": output_samples_keys, "formats": [float]*len(output_samples_keys)})
    output_samples = np.rec.fromarrays(list(output_samples.values()), dtype=output_samples_dt)
    f.create_dataset("posterior_samples", data=output_samples)
    f.attrs["log_Bayes_factor"] = log_Bayes_factor
    f.attrs["log_coherence_ratio"] = log_coherence_ratio
    f.close()

