import numpy as np
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.core.likelihood import Likelihood
from ...lensing.waveform import morse_phase_from_image_type

class ConditionalLikelihood(Likelihood):
    def __init__(self, parameters={}, trigger_ids=None, common_parameters=None, independent_parameters=None, base_posterior_samples=None, likelihood=None):
        super(ConditionalLikelihood, self).__init__(parameters=parameters)

        self.trigger_ids = trigger_ids
        self.common_parameters = common_parameters
        self.independent_parameters = independent_parameters
        self.base_posterior_samples = base_posterior_samples
        self.likelihood = likelihood

    def log_likelihood(self):
        ind = int(self.parameters["ind"])
        log_Z_cond = self.base_posterior_samples.iloc[ind]["log_conditional_evidence"]

        # Update the likelihood parameter
        # Common parameters from base posterior samples
        self.likelihood.parameters.update(self.base_posterior_samples.iloc[ind][self.common_parameters].to_dict())
        # The independent parameters
        # NOTE for JointLikelihood it assumes that the superscript starts from 0
        # This means we will have to rename the parameters
        for p in self.independent_parameters:
            for new_trigger_idx, old_trigger_idx in enumerate(self.trigger_ids[1:]):
                self.likelihood.parameters[p+self.likelihood.suffix(new_trigger_idx)] = \
                    self.parameters[p+self.likelihood.suffix(old_trigger_idx)]

        return self.likelihood.log_likelihood() - log_Z_cond


class SingleLikelihoodWithTransformableWaveformCache(GravitationalWaveTransient):
    @classmethod
    def from_likelihood(cls, likelihood, time_marginalization=False, distance_marginalization=False, distance_marginalization_lookup_table=None):
        lh = cls(
            likelihood.interferometers, likelihood.waveform_generator, time_marginalization=time_marginalization,
            distance_marginalization=distance_marginalization, phase_marginalization=False, priors=likelihood.priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table, jitter_time=False,
            reference_frame="sky", time_reference="geocent",
        )
        lh._cache = None
        return lh

    def initialize_cache(self, waveform_cache, parameters):
        self._cache = waveform_cache
        self._cache["waveform_parameters"] = {k:v for k,v in parameters.items() if not k.startswith("recalib_")}

    def log_likelihood_ratio(self, use_cache=True):
        if use_cache and self._cache is not None:
            self.transform_from_cache(self.parameters)
        return super(SingleLikelihoodWithTransformableWaveformCache, self).log_likelihood_ratio()

    def transform_from_cache(self, parameters):
        # Check if the cached waveform can be transformed to the desired waveform
        # The following parameters can be (and should be) different
        # FIXME This will break if sampled in detector-frame time
        parameters_excluded_from_comparison = ["image_type", "luminosity_distance", "geocent_time", "ra", "dec", "psi"]
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
            self.waveform_generator._cache = self._cache
