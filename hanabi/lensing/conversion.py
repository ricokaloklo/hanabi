from bilby.gw.conversion import *

def apparent_luminosity_distances_to_relative_magnification(
    apparent_luminosity_distance_1,
    apparent_luminosity_distance_2
):
    """
    \mu_{rel} = \mu^{(2)}_{abs} / \mu^{(1)}_{abs} = (d_L^(1) / d_L^(2))^2
    """
    return (apparent_luminosity_distance_1/apparent_luminosity_distance_2)**2

def convert_to_lal_binary_black_hole_parameters_for_lensed_BBH(parameters):
    """
    This prevents redshift being converted from luminosity distance (which is wrong)
    or luminosity distance being converted from redshift (which is wrong) when the signals
    are strongly lensed

    The proper conversion is done instead in LensingJointLikelihood.assign_trigger_level_parameters()
    """

    converted_parameters = parameters.copy()
    original_keys = list(converted_parameters.keys())

    for key in original_keys:
        if key[-7:] == '_source':
            converted_parameters[key[:-7]] = converted_parameters[key] * (
                1 + converted_parameters['redshift'])

    if 'chirp_mass' in converted_parameters.keys():
        if "mass_1" in converted_parameters.keys():
            converted_parameters["mass_ratio"] = chirp_mass_and_primary_mass_to_mass_ratio(
                converted_parameters["chirp_mass"], converted_parameters["mass_1"])
        if 'total_mass' in converted_parameters.keys():
            converted_parameters['symmetric_mass_ratio'] =\
                chirp_mass_and_total_mass_to_symmetric_mass_ratio(
                    converted_parameters['chirp_mass'],
                    converted_parameters['total_mass'])
        if 'symmetric_mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_ratio'] =\
                symmetric_mass_ratio_to_mass_ratio(
                    converted_parameters['symmetric_mass_ratio'])
        if 'total_mass' not in converted_parameters.keys():
            converted_parameters['total_mass'] =\
                chirp_mass_and_mass_ratio_to_total_mass(
                    converted_parameters['chirp_mass'],
                    converted_parameters['mass_ratio'])
        converted_parameters['mass_1'], converted_parameters['mass_2'] = \
            total_mass_and_mass_ratio_to_component_masses(
                converted_parameters['mass_ratio'],
                converted_parameters['total_mass'])
    elif 'total_mass' in converted_parameters.keys():
        if 'symmetric_mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_ratio'] = \
                symmetric_mass_ratio_to_mass_ratio(
                    converted_parameters['symmetric_mass_ratio'])
        if 'mass_ratio' in converted_parameters.keys():
            converted_parameters['mass_1'], converted_parameters['mass_2'] =\
                total_mass_and_mass_ratio_to_component_masses(
                    converted_parameters['mass_ratio'],
                    converted_parameters['total_mass'])
        elif 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['total_mass'] -\
                converted_parameters['mass_1']
        elif 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] = \
                converted_parameters['total_mass'] - \
                converted_parameters['mass_2']
    elif 'symmetric_mass_ratio' in converted_parameters.keys():
        converted_parameters['mass_ratio'] =\
            symmetric_mass_ratio_to_mass_ratio(
                converted_parameters['symmetric_mass_ratio'])
        if 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['mass_1'] *\
                converted_parameters['mass_ratio']
        elif 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] =\
                converted_parameters['mass_2'] /\
                converted_parameters['mass_ratio']
    elif 'mass_ratio' in converted_parameters.keys():
        if 'mass_1' in converted_parameters.keys():
            converted_parameters['mass_2'] =\
                converted_parameters['mass_1'] *\
                converted_parameters['mass_ratio']
        if 'mass_2' in converted_parameters.keys():
            converted_parameters['mass_1'] = \
                converted_parameters['mass_2'] /\
                converted_parameters['mass_ratio']

    for idx in ['1', '2']:
        key = 'chi_{}'.format(idx)
        if key in original_keys:
            converted_parameters['a_{}'.format(idx)] = abs(
                converted_parameters[key])
            converted_parameters['cos_tilt_{}'.format(idx)] = \
                np.sign(converted_parameters[key])
            converted_parameters['phi_jl'] = 0.0
            converted_parameters['phi_12'] = 0.0

    for angle in ['tilt_1', 'tilt_2', 'theta_jn']:
        cos_angle = str('cos_' + angle)
        if cos_angle in converted_parameters.keys():
            converted_parameters[angle] =\
                np.arccos(converted_parameters[cos_angle])

    if "delta_phase" in original_keys:
        converted_parameters["phase"] = np.mod(
            converted_parameters["delta_phase"]
            - np.sign(np.cos(converted_parameters["theta_jn"]))
            * converted_parameters["psi"],
            2 * np.pi
        )

    added_keys = [key for key in converted_parameters.keys()
                  if key not in original_keys]

    return converted_parameters, added_keys
