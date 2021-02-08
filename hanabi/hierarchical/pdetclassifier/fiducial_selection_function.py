import numpy as np
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

__prog__ = "fiducial_selection_function.py"
setup_logger(__prog__)
logger = logging.getLogger(__prog__)

label = "o3a_first3months_bbhpop_fiducial"
trained_model = "trained_2e7_O3_precessing_higherordermodes_3detectors.h5"

seed = 12345
if seed is not None:
	logger.info(f"Setting random seed as {seed}")
	np.random.seed(seed)

N = int(1e6)
N_z = 100

class Fixed(object):
	def __init__(self, fixed_value):
		self.fixed_value = fixed_value

	def sample(self, size=1):
		return np.ones(size)*self.fixed_value

	def prob(self, value):
		return 1*(value == self.fixed_value)

# Setting up the population and lensing model
mmin = 5
mmax = 41
intrinsic_pop_models = {
	"m1_src": bilby.core.prior.Uniform(minimum=mmin, maximum=mmax),
	"m2_src": bilby.core.prior.Uniform(minimum=mmin, maximum=mmax),
	"chi1x": Fixed(0.0),
	"chi1y": Fixed(0.0),
	"chi1z": bilby.core.prior.Uniform(minimum=-1,maximum=1),
	"chi2x": Fixed(0.0),
	"chi2y": Fixed(0.0),
	"chi2z": bilby.core.prior.Uniform(minimum=-1,maximum=1),
}
lensing_models = {
	"mu": hanabi.lensing.absolute_magnification.SISPowerLawAbsoluteMagnification(),
}
merger_rate_density = hanabi.hierarchical.merger_rate_density.BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity()
optical_depth = hanabi.lensing.optical_depth.HannukselaEtAl2019OpticalDepth()
pz_lensed = hanabi.hierarchical.p_z.LensedSourceRedshiftProbDist(merger_rate_density=merger_rate_density, optical_depth=optical_depth)
pz_notlensed = hanabi.hierarchical.p_z.NotLensedSourceRedshiftProbDist(merger_rate_density=merger_rate_density, optical_depth=optical_depth)

def generate_intrinsic_parameters(N):
	# Masses
	m1_src = intrinsic_pop_models["m1_src"].sample(size=N)
	m2_src = intrinsic_pop_models["m2_src"].sample(size=N)
	q = np.zeros_like(m1_src)
	for idx, (m1, m2) in enumerate(zip(m1_src, m2_src)):
		if m1 > m2:
			q[idx] = m2/m1
		else:
			q[idx] = m1/m2
	pdf_mass = np.ones_like(m1_src)*(intrinsic_pop_models["m1_src"].prob(m1_src))*(intrinsic_pop_models["m2_src"].prob(m2_src))

	# Spins
	chi1x = intrinsic_pop_models["chi1x"].sample(size=N)
	chi1y = intrinsic_pop_models["chi1y"].sample(size=N)
	chi1z = intrinsic_pop_models["chi1z"].sample(size=N)
	chi2x = intrinsic_pop_models["chi2x"].sample(size=N)
	chi2y = intrinsic_pop_models["chi2y"].sample(size=N)
	chi2z = intrinsic_pop_models["chi2z"].sample(size=N)
	pdf_spin = np.ones_like(chi1z)* \
		(intrinsic_pop_models["chi1x"].prob(chi1x))* \
		(intrinsic_pop_models["chi1y"].prob(chi1y))* \
		(intrinsic_pop_models["chi1z"].prob(chi1z))* \
		(intrinsic_pop_models["chi2x"].prob(chi2x))* \
		(intrinsic_pop_models["chi2y"].prob(chi2y))* \
		(intrinsic_pop_models["chi2z"].prob(chi2z))
	
	return m1_src, m2_src, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, pdf_mass, pdf_spin

# Setting up a fiducial population
binaries = pdetclassifier.generate_binaries(N)
m1_src, m2_src, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, pdf_mass, pdf_spin = generate_intrinsic_parameters(N)
binaries['chi1x'] = chi1x
binaries['chi1y'] = chi1y
binaries['chi1z'] = chi1z
binaries['chi2x'] = chi2x
binaries['chi2y'] = chi2y
binaries['chi2z'] = chi2z

# Redshift
zs = pz_notlensed.sample(size=N)
pdf_z = pz_notlensed.prob(zs)

binaries['mtot'] = (m1_src + m2_src)*(1 + zs)
binaries['q'] = q
binaries['z'] = zs

logger.info(f"Calculating the BBH selection function alpha")
# Calculating the BBH selection function alpha
model = pdetclassifier.loadnetwork(trained_model)
predictions = pdetclassifier.predictnetwork(model, binaries)
integral = np.sum(predictions).astype(float)/N # Monte-Carlo integration

# Writing the results to a HDF5 file
output_binaries = {"m1_src": m1_src, "m2_src": m2_src, "sampling_pdf": pdf_mass+pdf_spin+pdf_z}
output_binaries.update(binaries)
del output_binaries["N"]
output_binaries_keys = list(output_binaries.keys())
output_binaries_dt = np.dtype({"names": output_binaries_keys, "formats": [float]*len(output_binaries_keys)})
output_binaries = np.rec.array([output_binaries[k] for k in output_binaries_keys], dtype=output_binaries_dt)
output_filename = "{}_alpha.h5".format(label)
logger.info(f"Writing output to {output_filename}")
with h5py.File(output_filename, "w") as f:
	dataset = f.create_dataset("binaries", data=output_binaries)
	f.attrs["N"] = N
	f.attrs["alpha"] = integral


logger.info(f"Building a fast interpolant from luminosity distance to redshift")
# Building a fast interpolant for dL -> redshift
z_fit_max = merger_rate_density.population_parameter_dict["redshift_max"]
zs_int = np.linspace(0, z_fit_max, num=100)
dL_int = bilby.gw.conversion.redshift_to_luminosity_distance(zs_int)
spline_int = scipy.interpolate.splrep(dL_int, zs_int, s=0)

# Setting up a fiducial population model
binaries = {} # New dictionary
binaries = pdetclassifier.generate_binaries(N)

m1_src, m2_src, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, pdf_mass, pdf_spin = generate_intrinsic_parameters(N)
binaries['chi1x'] = chi1x
binaries['chi1y'] = chi1y
binaries['chi1z'] = chi1z
binaries['chi2x'] = chi2x
binaries['chi2y'] = chi2y
binaries['chi2z'] = chi2z

zs = pz_lensed.sample(size=N_z)
pdf_z = pz_lensed.prob(zs)
inner_integral = []

# Prepare a handle for HDF5 IO
output_filename = "{}_beta.h5".format(label)
f = h5py.File(output_filename, "w")
output_binaries = {"m1_src": m1_src, "m2_src": m2_src, "sampling_pdf": pdf_mass+pdf_spin}
output_binaries.update(binaries)
del output_binaries["N"]
output_binaries_keys = list(output_binaries.keys())
output_binaries_dt = np.dtype({"names": output_binaries_keys, "formats": [float]*len(output_binaries_keys)})
output_binaries = np.rec.array([output_binaries[k] for k in output_binaries_keys], dtype=output_binaries_dt)
f.create_dataset("binaries", data=output_binaries)

logger.info(f"Calculating the lensing selection function beta")
# Loop over zs
for z_src in tqdm.tqdm(zs):
	# Calculating the inner integral using Monte Carlo method
	dL_src = bilby.gw.conversion.redshift_to_luminosity_distance(z_src)
	abs_magn_1 = lensing_models["mu"].sample(size=N)
	abs_magn_2 = lensing_models["mu"].sample(size=N)
	dL_1_apparent = dL_src/np.sqrt(abs_magn_1)
	dL_2_apparent = dL_src/np.sqrt(abs_magn_2)
	z_1_apparent = scipy.interpolate.splev(dL_1_apparent, spline_int, der=0)
	z_2_apparent = scipy.interpolate.splev(dL_2_apparent, spline_int, der=0)

	binaries['mtot'] = (m1_src + m2_src)*(1 + z_src)
	binaries['q'] = q
	binaries['z'] = z_1_apparent

	binaries_2 = copy.deepcopy(binaries)
	binaries_2['z'] = z_2_apparent

	# Integrating over entire population
	predictions = pdetclassifier.predictnetwork(model, binaries)
	predictions_2 = pdetclassifier.predictnetwork(model, binaries_2)
	integral = np.sum(predictions*predictions_2).astype(float)/N
	inner_integral.append(integral)

	# Storing extra information
	binaries['apparent_dL'] = dL_1_apparent
	binaries['abs_magn'] = abs_magn_1
	binaries_2['apparent_dL'] = dL_2_apparent
	binaries_2['abs_magn'] = abs_magn_2

beta = np.sum(inner_integral).astype(float)/N_z # Monte Carlo integration

logger.info(f"Writing output to {output_filename}")
# Writing the results to a HDF5 file
output_zs_dt = np.dtype({"names": ["z", "sampling_pdf"], "formats": [float]*2})
output_zs = np.rec.array([zs, pdf_z], dtype=output_zs_dt)
f.create_dataset("zs", data=output_zs)
f.create_dataset("inner_integral", data=inner_integral)
f.attrs["N"] = N
f.attrs["N_z"] = N_z
f.attrs["beta"] = beta
f.close()
