import numpy as np
from bilby.gw.source import *

def strongly_lensed_BBH_waveform(
    frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, image_type, **kwargs
):
    frequency_domain_source_model = lal_binary_black_hole
    image_type = int(image_type)

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
    wf = frequency_domain_source_model(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs
    )

    # Apply a phase shift per polarization
    wf["plus"] = np.exp(-1j*phase_shift_dict[image_type])*np.ones_like(wf["plus"]) * wf["plus"]
    wf["cross"] = np.exp(-1j*phase_shift_dict[image_type])*np.ones_like(wf["cross"]) * wf["cross"]

    return wf
