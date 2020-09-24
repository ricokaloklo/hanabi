import numpy as np
from bilby.gw.source import *

def strongly_lensed_BBH_waveform(**kwargs):
    frequency_domain_source_model = kwargs.pop("frequency_domain_source_model", lal_binary_black_hole) # Default is lal_binary_black_hole
    image_type = int(kwargs.pop("image_type", 1)) # Default is type-I (no phase shift)

    """
    Type-I image: 0 phase shift
    Type-II image: pi/2 phase shift
    Type-III image: pi phase shift
    """
    phase_shift_dict = {
        1: 0,
        2: np.pi/2.,
        3: np.pi
    }

    # Actually generate the waveform by calling the generator
    wf = frequency_domain_source_model(**kwargs)

    # Apply a phase shift per polarization
    wf["plus"] = np.exp(1j*phase_shift_dict[image_type])*np.ones_like(wf["plus"]) * wf["plus"]
    wf["cross"] = np.exp(1j*phase_shift_dict[image_type])*np.ones_like(wf["cross"]) * wf["cross"]

    return wf
