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
    def __init__(self, single_trigger_likelihoods, sep_char="^", suffix=None):
        if suffix is None:
            suffix = ParameterSuffix(sep_char)
        super(LensingJointLikelihood, self).__init__(single_trigger_likelihoods, sep_char=sep_char, suffix=suffix)

        # FIXME For now we assign convert_to_lal_binary_black_hole_parameters_for_lensed_BBH as the default parameter_conversion function
        for single_trigger_likelihood in self.single_trigger_likelihoods:
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


class LensingJointLikelihoodWithWaveformCache(LensingJointLikelihood):
    def initialize_cache(self):
        self._cache = dict(parameters=None, waveform=None, model=None)
    
    def add_to_cache(self, likelihood, parameters):
        self._cache = copy.deepcopy(likelihood.waveform_generator._cache)
        self._cache["waveform_parameters"] = {k:v for k,v in parameters.items() if not k.startswith("recalib_")}

    def transform_from_cache(self, likelihood, parameters):
        # Check if the cached waveform can be transformed to the desired waveform
        # The following parameters can be (and should be) different
        # FIXME This will break if sampled in detector-frame time
        parameters_excluded_from_comparison = ["image_type", "luminosity_distance", "geocent_time"]
        parameters_to_be_checked = [k for k in parameters.keys() if k not in parameters_excluded_from_comparison and not k.startswith("recalib_")]
        if {k:v for k,v in parameters.items() if k in parameters_to_be_checked} == {k:v for k,v in self._cache["waveform_parameters"].items() if k in parameters_to_be_checked}:
            scale = self._cache["waveform_parameters"]["luminosity_distance"]/parameters["luminosity_distance"]
            phase = morse_phase_from_image_type(int(parameters["image_type"])) - morse_phase_from_image_type(int(self._cache["waveform_parameters"]["image_type"]))

            # Perform the transformation
            self._cache["waveform"]["plus"] = np.exp(-1j*phase)*np.ones_like(self._cache["waveform"]["plus"])*scale*self._cache["waveform"]["plus"]
            self._cache["waveform"]["cross"] = np.exp(-1j*phase)*np.ones_like(self._cache["waveform"]["cross"])*scale*self._cache["waveform"]["cross"]

            self._cache["parameters"]["image_type"] = parameters["image_type"]
            self._cache["parameters"]["luminosity_distance"] = parameters["luminosity_distance"]

            # Assign this cache to the waveform generator
            likelihood.waveform_generator._cache = self._cache


    def log_likelihood(self):
        # Sum over all the log_likelihood values with the appropriate parameters passed
        parameters_per_trigger = self.assign_trigger_level_parameters(full_parameters=self.parameters)
        logL = 0.0

        # Initialize cache per joint likelihood evaluation
        self.initialize_cache()

        for single_trigger_likelihood, single_trigger_parameters in zip(self.single_trigger_likelihoods, parameters_per_trigger):
            if self._cache["waveform"] is not None:
                self.transform_from_cache(single_trigger_likelihood, single_trigger_parameters)

            # Assign the single_trigger_parameters to the likelihood object for evaluation
            single_trigger_likelihood.parameters.update(single_trigger_parameters)

            # Calculate the log likelihood
            logL += single_trigger_likelihood.log_likelihood()

            # Update cache if empty
            if self._cache["waveform"] is None:
                self.add_to_cache(single_trigger_likelihood, single_trigger_parameters)

        return logL