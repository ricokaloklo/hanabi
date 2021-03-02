import numpy as np
import pandas as pd
import h5py
import copy
import scipy.interpolate
import tqdm
import pdetclassifier
import bilby
import hanabi.hierarchical
import hanabi.lensing
import logging
from hanabi.inference.utils import setup_logger

__prog__ = "compute_selection_functions"
setup_logger(__prog__)
logger = logging.getLogger(__prog__)

label = "o3a_first3months_bbhpop_powerlaw_mmax_60_isotropic_spin"
trained_model = "trained_2e7_O3_precessing_higherordermodes_3detectors.h5"

seed = 12345
if seed is not None:
	logger.info(f"Setting random seed as {seed}")
	np.random.seed(seed)

N = int(1e7)
N_z = 100

class Fixed(object):
	def __init__(self, fixed_value):
		self.fixed_value = fixed_value

	def sample(self, size=1):
		return np.ones(size)*self.fixed_value

	def prob(self, value):
		return 1*(value == self.fixed_value)

# Setting up the population and lensing model
mass_src_pop_model = hanabi.hierarchical.source_population_model.PowerLawComponentMass(alpha=1.8, beta=0, mmin=5, mmax=60)
spin_src_pop_model = None # Currently we use the default uniform-magnitude-isotropic-spin distribution in pdetclassifier
abs_magn_dist = hanabi.lensing.absolute_magnification.SISPowerLawAbsoluteMagnification()
merger_rate_density = hanabi.hierarchical.merger_rate_density.BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity()
optical_depth = hanabi.lensing.optical_depth.HannukselaEtAl2019OpticalDepth()
pz_lensed = hanabi.hierarchical.p_z.LensedSourceRedshiftProbDist(merger_rate_density=merger_rate_density, optical_depth=optical_depth)
pz_notlensed = hanabi.hierarchical.p_z.NotLensedSourceRedshiftProbDist(merger_rate_density=merger_rate_density, optical_depth=optical_depth)

# Fiducial intrinsic population model. We generate samples from these distribution instead
fiducial_intrinsic_pop_models = {
	"m1_src": bilby.core.prior.Uniform(minimum=5, maximum=100),
	"m2_src": bilby.core.prior.Uniform(minimum=5, maximum=100),
}

def generate_intrinsic_parameters(N):
	# Masses
	m1_src = fiducial_intrinsic_pop_models["m1_src"].sample(size=N)
	m2_src = fiducial_intrinsic_pop_models["m2_src"].sample(size=N)
	for idx, (m1, m2) in enumerate(zip(m1_src, m2_src)):
		if m1 < m2:
			tmp = m1
			m1_src[idx] = m2
			m2_src[idx] = tmp
	q = m2_src/m1_src
	pdf_mass = np.ones_like(m1_src)*(fiducial_intrinsic_pop_models["m1_src"].prob(m1_src))*(fiducial_intrinsic_pop_models["m2_src"].prob(m2_src))
	
	return m1_src, m2_src, q, pdf_mass

# Setting up a fiducial population
binaries = pdetclassifier.generate_binaries(N)
m1_src, m2_src, q, pdf_mass = generate_intrinsic_parameters(N)
weights_mass = mass_src_pop_model.prob(pd.DataFrame({"mass_1_source": m1_src, "mass_2_source": m2_src}), axis=0)/pdf_mass
weights = weights_mass

# Redshift
zs_notlensed = pz_notlensed.sample(size=N)
pdf_z_notlensed = pz_notlensed.prob(zs_notlensed)

binaries['mtot'] = (m1_src + m2_src)*(1 + zs_notlensed)
binaries['q'] = q
binaries['z'] = zs_notlensed

logger.info(f"Calculating the BBH selection function alpha")
# Calculating the BBH selection function alpha
model = pdetclassifier.loadnetwork(trained_model)
predictions = pdetclassifier.predictnetwork(model, binaries)
alpha = np.sum(predictions*weights).astype(float)/N # Monte-Carlo importance-weighted integration


logger.info(f"Building a fast interpolant from luminosity distance to redshift")
# Building a fast interpolant for dL -> redshift
z_fit_max = merger_rate_density.population_parameter_dict["redshift_max"]
zs_int = np.linspace(0, z_fit_max, num=100)
dL_int = bilby.gw.conversion.redshift_to_luminosity_distance(zs_int)
spline_int = scipy.interpolate.splrep(dL_int, zs_int, s=0)

# Setting up a fiducial population model
binaries = {} # New dictionary
binaries = pdetclassifier.generate_binaries(N)

zs_lensed = pz_lensed.sample(size=N_z)
pdf_z_lensed = pz_lensed.prob(zs_lensed)
inner_integral = []

logger.info(f"Calculating the lensing selection function beta")
# Loop over zs
for z_src in tqdm.tqdm(zs_lensed):
	# Calculating the inner integral using Monte Carlo method
	dL_src = bilby.gw.conversion.redshift_to_luminosity_distance(z_src)
	abs_magn_1 = abs_magn_dist.sample(size=N)
	abs_magn_2 = abs_magn_dist.sample(size=N)
	dL_1_apparent = dL_src/np.sqrt(abs_magn_1)
	dL_2_apparent = dL_src/np.sqrt(abs_magn_2)
	z_1_apparent = scipy.interpolate.splev(dL_1_apparent, spline_int, der=0)
	z_2_apparent = scipy.interpolate.splev(dL_2_apparent, spline_int, der=0)

	# Using the m1_src, m2_src drawn from the fiducial distributions
	binaries['mtot'] = (m1_src + m2_src)*(1 + z_src)
	binaries['q'] = q
	binaries['z'] = z_1_apparent

	binaries_2 = copy.deepcopy(binaries)
	binaries_2['z'] = z_2_apparent

	# Integrating over entire population
	predictions = pdetclassifier.predictnetwork(model, binaries)
	predictions_2 = pdetclassifier.predictnetwork(model, binaries_2)
	integral = np.sum(predictions*predictions_2*weights).astype(float)/N
	inner_integral.append(integral)

inner_integral = np.array(inner_integral)
beta = np.sum(inner_integral).astype(float)/N_z # Monte Carlo integration

# Prepare a handle for HDF5 IO
output_filename = "{}.h5".format(label)
f = h5py.File(output_filename, "w")
# Writing the results to a HDF5 file
logger.info(f"Writing output to {output_filename}")
output_binaries = {"mass_1_source": m1_src, "mass_2_source": m2_src, "spin_1x": binaries["chi1x"], "spin_1y": binaries["chi1y"], "spin_1z": binaries["chi1z"], "spin_2x": binaries["chi2x"], "spin_2y": binaries["chi2y"], "spin_2z": binaries["chi2z"], "sampling_pdf": pdf_mass, "weight": weights}
output_binaries_keys = list(output_binaries.keys())
output_binaries_dt = np.dtype({"names": output_binaries_keys, "formats": [float]*len(output_binaries_keys)})
output_binaries = np.rec.array(list(output_binaries.values()), dtype=output_binaries_dt)
output_inner_integral_dt = np.dtype({"names": ["z", "epsilon", "sampling_pdf"], "formats": [float]*3})
output_inner_integral = np.rec.array([zs_lensed, inner_integral, pdf_z_lensed], dtype=output_inner_integral_dt)
f.create_dataset("binaries", data=output_binaries)
f.create_dataset("epsilon", data=output_inner_integral)
f.attrs["N"] = N
f.attrs["N_z"] = N_z
f.attrs["alpha"] = alpha
f.attrs["beta"] = beta
f.close()