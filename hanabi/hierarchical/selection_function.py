import os
import numpy as np
import h5py
import pandas as pd
from .source_population_model import Marginalized
from .merger_rate_density import MarginalizedMergerRateDensity

class SelectionFunction(object):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity()
    ):
        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        self.merger_rate_density_src_pop_model = merger_rate_density_src_pop_model
    
    def expected_number_of_mergers(self, T_obs, IFAR_threshold=1):
        return 1.0

    def evaluate(self, IFAR_threshold=1):
        # T_obs does not matter in evaluating the selection function
        N_tot = self.merger_rate_density_src_pop_model.total_number_of_mergers(1.0)
        N_exp = self.expected_number_of_mergers(1.0, IFAR_threshold=IFAR_threshold)

        return N_exp/N_tot


class BinaryBlackHoleSelectionFunctionFromInjection(SelectionFunction):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity(),
        filename=None
    ):
        super(BinaryBlackHoleSelectionFunctionFromInjection, self).__init__(
            mass_src_pop_model=mass_src_pop_model,
            spin_src_pop_model=spin_src_pop_model,
            merger_rate_density_src_pop_model=merger_rate_density_src_pop_model
        )

        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "data", "o3a_bbhpop_inj_info.hdf")
        pop_inj_info = h5py.File(filename, "r")
        # Read in dataset and attributes from the file
        for key, value in pop_inj_info.attrs.items():
            if key == "N_exp/R(z=0)":
                # This particular key does not play well
                key = "N_exp_over_R_0"
            setattr(self, "pop_inj_{}".format(key), value)

        column_names = list(pop_inj_info["injections"])
        self.pop_inj_info = pd.DataFrame(data={column: pop_inj_info["injections"][column] for column in column_names})

        pop_inj_info.close()

        self.pop_inj_analysis_time_yr = self.pop_inj_analysis_time_s/(365.25*24*3600)
        # Rename the column
        self.pop_inj_info.rename(columns={
            "mass1_source": "mass_1_source",
            "mass2_source": "mass_2_source",
            "spin1z": "spin_1z",
            "spin2z": "spin_2z",
        }, inplace=True)

    def expected_number_of_mergers(self, T_obs, IFAR_threshold=1):
        # Default IFAR threshold= 1 yr
        # Detected either by gstlal, pycbc_full or pycbc_bbh
        detected = (self.pop_inj_info["ifar_gstlal"] > IFAR_threshold) | (self.pop_inj_info["ifar_pycbc_full"] > IFAR_threshold) | (self.pop_inj_info["ifar_pycbc_bbh"] > IFAR_threshold)

        # Actually compute the log of expected number
        log_dN = np.where(
            detected,
            self.mass_src_pop_model.ln_prob(self.pop_inj_info[["mass_1_source", "mass_2_source"]], axis=0) + \
            self.spin_src_pop_model.ln_prob(self.pop_inj_info[["spin_1z", "spin_2z"]], axis=0) + \
            self.merger_rate_density_src_pop_model.ln_dN_over_dz(self.pop_inj_info),
            np.NINF
        )
        log_p_draw = np.log(self.pop_inj_info["sampling_pdf"].to_numpy())
        log_N_exp = np.log(T_obs) + np.logaddexp.reduce(log_dN - log_p_draw) - np.log(self.pop_inj_total_generated)

        return np.exp(log_N_exp)

