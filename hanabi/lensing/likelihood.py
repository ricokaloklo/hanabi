import numpy as np
import copy
import bilby.core.likelihood
import bilby.gw.conversion
import bilby.gw.likelihood
from .waveforms import *

class LensingJointLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, single_trigger_likelihoods, lensed_waveform_model, sep_char="^", suffix=None):
        # Initialize some variables using the default constructor
        super(LensingJointLikelihood, self).__init__(parameters={})
        self.single_trigger_likelihoods = single_trigger_likelihoods
        self.n_triggers = len(self.single_trigger_likelihoods) # Reconstruct the number of triggers
        self.lensed_waveform_model = lensed_waveform_model

        # Assign the lensed waveform model specified to the single-trigger likelihoods
        for single_trigger_likelihood in self.single_trigger_likelihoods:
            single_trigger_likelihood.waveform_generator.frequency_domain_source_model = self.lensed_waveform_model

        self.sep_char = sep_char
        if suffix is None:
            self.suffix = lambda trigger_idx: "{}({})".format(self.sep_char, trigger_idx + 1)
        else:
            self.suffix = suffix

    def assign_trigger_level_parameters(self, full_parameters=None):
        if full_parameters is None:
            full_parameters = self.parameters
        
        common_parameters = [p for p in full_parameters.keys() if self.sep_char not in p]
        parameters_per_trigger = []

        # Reconstruct the set of parameters that should have been passed to the likelihood
        for trigger_idx in range(self.n_triggers):
            suffix = self.suffix(trigger_idx)

            # Remove/slaughter parameters
            name_mappings = {}
            for param in full_parameters.keys():
                if self.sep_char in param:
                    if suffix in param:
                        name_mappings[param] = param.replace(suffix, "") # Rename the parameter to the correct one
                    else:
                        name_mappings[param] = "" # Remove the parameter
                else:
                    name_mappings[param] = param # Keep this parameter

            # Construct a new dict
            trigger_parameters = {}
            for old_name, new_name in name_mappings.items():
                if new_name == "":
                    continue
                else:
                    # In theory the object stored should be immutable, but just to be safe
                    trigger_parameters[new_name] = copy.deepcopy(full_parameters[old_name])

            # Handle cases for lensing magnification
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
                    trigger_parameters["luminosity_distance"] = source_luminosity_distance/np.sqrt(absolute_magnification)
                    # Convert source luminosity distance to source redshift in case needed
                    trigger_parameters["redshift"] = bilby.gw.conversion.luminosity_distance_to_redshift(source_luminosity_distance)

            if "redshift" in common_parameters:
                raise NotImplementedError("The code does not support sampling over redshift yet")

            parameters_per_trigger.append(trigger_parameters)
        
        return parameters_per_trigger

    def log_likelihood(self):
        # Sum over all the log_likelihood values with the appropriate parameters passed
        parameters_per_trigger = self.assign_trigger_level_parameters()
        logL = 0.0

        for single_trigger_likelihood, single_trigger_parameters in zip(self.single_trigger_likelihoods, parameters_per_trigger):
            # Assign the single_trigger_parameters to the likelihood object for evaluation
            single_trigger_likelihood.parameters = single_trigger_parameters

            # Calculate the log likelihood
            logL += single_trigger_likelihood.log_likelihood()

        return logL

    def noise_log_likelihood(self):
        # Sum over all the noise_log_likelihood values
        return np.sum([single_trigger_likelihood.noise_log_likelihood() for single_trigger_likelihood in self.single_trigger_likelihoods])