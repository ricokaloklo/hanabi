import numpy as np
from bilby.gw.source import *

def phase_shifted_per_polarization_BBH(
    phase_shift, frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):
    
    wf = lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    # Apply a phase shift per polarization
    wf["plus"] = np.exp(1j*phase_shift)*np.ones_like(wf["plus"]) * wf["plus"]
    wf["cross"] = np.exp(1j*phase_shift)*np.ones_like(wf["cross"]) * wf["cross"]

    return wf

def lensed_BBH_type_I_image(
    frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    return phase_shifted_per_polarization_BBH(
    0.0, frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

def lensed_BBH_type_II_image(
    frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    return phase_shifted_per_polarization_BBH(
    np.pi/2., frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

def lensed_BBH_type_III_image(
    frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    return phase_shifted_per_polarization_BBH(
    np.pi, frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
    phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
