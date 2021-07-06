import numpy as np
import copy
import bilby.core.likelihood
import bilby.gw.conversion
import bilby.gw.likelihood
from ..inference.likelihood import JointLikelihood
from ..inference.utils import ParameterSuffix
from .waveform import *
from .conversion import convert_to_lal_binary_black_hole_parameters_for_lensed_BBH

class LensingJointLikelihood(JointLikelihood):
    def __init__(self, single_trigger_likelihoods, lensed_waveform_model, sep_char="^", suffix=None):
        if suffix is None:
            suffix = ParameterSuffix(sep_char)
        super(LensingJointLikelihood, self).__init__(single_trigger_likelihoods, sep_char=sep_char, suffix=suffix)
        self.lensed_waveform_model = lensed_waveform_model

        # Assign the lensed waveform model specified to the single-trigger likelihoods
        # FIXME For now we assign convert_to_lal_binary_black_hole_parameters_for_lensed_BBH as the default parameter_conversion function
        for single_trigger_likelihood in self.single_trigger_likelihoods:
            single_trigger_likelihood.waveform_generator.frequency_domain_source_model = self.lensed_waveform_model
            single_trigger_likelihood.waveform_generator.parameter_conversion = convert_to_lal_binary_black_hole_parameters_for_lensed_BBH

    def assign_trigger_level_parameters(self, full_parameters=None):
        if full_parameters is None:
            full_parameters = self.parameters

        common_parameters = [p for p in full_parameters.keys() if self.sep_char not in p]
        parameters_per_trigger = super(LensingJointLikelihood, self).assign_trigger_level_parameters(full_parameters)

        # Handle cases for lensing magnification
        for trigger_idx in range(self.n_triggers):
            trigger_parameters = parameters_per_trigger[trigger_idx]

            if "luminosity_distance" in common_parameters:
                if "relative_magnification" in trigger_parameters.keys():
                    # Overwrite the luminosity distance here to the apparent luminosity distance
                    relative_magnification = trigger_parameters.pop("relative_magnification")
                    trigger_parameters["luminosity_distance"] = trigger_parameters["luminosity_distance"]/(np.sqrt(relative_magnification))
                    # NOTE We cannot get the source redshift here!
                elif "absolute_magnification" in trigger_parameters.keys():
                    # Overwrite the luminosity distance here to the apparent luminosity distance
                    absolute_magnification = trigger_parameters.pop("absolute_magnification")
                    source_luminosity_distance = trigger_parameters.pop("luminosity_distance")
                    # Convert source luminosity distance to source redshift in case needed
                    trigger_parameters["redshift"] = bilby.gw.conversion.luminosity_distance_to_redshift(source_luminosity_distance).item()

                    trigger_parameters["luminosity_distance"] = source_luminosity_distance/np.sqrt(absolute_magnification)
            elif "redshift" in common_parameters:
                if "relative_magnification" in trigger_parameters.keys():
                    raise NotImplementedError("Sampling in redshift requires absolute magnification. Sample in luminosity distance instead if you want to use relative magnification")
                elif "absolute_magnification" in trigger_parameters.keys():
                    source_redshift = trigger_parameters["redshift"]
                    source_luminosity_distance = bilby.gw.conversion.redshift_to_luminosity_distance(source_redshift)
                    # Overwrite the luminosity distance here to the apparent luminosity distance
                    absolute_magnification = trigger_parameters.pop("absolute_magnification")
                    trigger_parameters["luminosity_distance"] = source_luminosity_distance/np.sqrt(absolute_magnification)
            elif "comoving_distance" in common_parameters:
                raise NotImplementedError("Currently do not support sampling in comoving distance")

        return parameters_per_trigger
