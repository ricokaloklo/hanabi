import numpy as np
import copy
import bilby.core.likelihood

class JointLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, single_trigger_likelihoods, sep_char="^", suffix=None):
        # Initialize some variables using the default constructor
        super(JointLikelihood, self).__init__(parameters={})
        self.single_trigger_likelihoods = single_trigger_likelihoods
        self.n_triggers = len(self.single_trigger_likelihoods) # Reconstruct the number of triggers

        self.sep_char = sep_char
        self.suffix = suffix

    def assign_trigger_level_parameters(self, full_parameters=None):
        if full_parameters is None:
            full_parameters = self.parameters
        
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

            parameters_per_trigger.append(trigger_parameters)
        
        return parameters_per_trigger

    def log_likelihood(self):
        # Sum over all the log_likelihood values with the appropriate parameters passed
        parameters_per_trigger = self.assign_trigger_level_parameters(full_parameters=self.parameters)
        logL = 0.0

        for single_trigger_likelihood, single_trigger_parameters in zip(self.single_trigger_likelihoods, parameters_per_trigger):
            # Assign the single_trigger_parameters to the likelihood object for evaluation
            single_trigger_likelihood.parameters.update(single_trigger_parameters)

            # Calculate the log likelihood
            logL += single_trigger_likelihood.log_likelihood()

        return logL

    def noise_log_likelihood(self):
        # Sum over all the noise_log_likelihood values
        return np.sum([single_trigger_likelihood.noise_log_likelihood() for single_trigger_likelihood in self.single_trigger_likelihoods])
