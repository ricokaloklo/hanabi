import numpy as np
import h5py
import pdetclassifier
import bilby
import hanabi.hierarchical
import hanabi.lensing

label = "o3a_first3months_bbhpop_fiducial"
trained_model = "trained_2e7_O3_precessing_higherordermodes_3detectors.h5"

# Setting up a fiducial population model
N = int(1e6)
binaries = pdetclassifier.generate_binaries(N)

# Masses
mmin = 5
mmax = 200
m1_src = bilby.core.prior.Uniform(minimum=mmin, maximum=mmax).sample(size=N)
m2_src = bilby.core.prior.Uniform(minimum=mmin, maximum=mmax).sample(size=N)
q = np.zeros_like(m1_src)
for idx, (m1, m2) in enumerate(zip(m1_src, m2_src)):
	if m1 > m2:
		q[idx] = m2/m1
	else:
		q[idx] = m1/m2
pdf_mass = np.ones_like(m1_src)*(1./(mmax-mmin))*(1./(mmax-mmin))

# Spins
chi1x = np.zeros(N)
chi1y = np.zeros(N)
chi1z = np.random.uniform(-1.0, 1.0, N)
chi2x = np.zeros(N)
chi2y = np.zeros(N)
chi2z = np.random.uniform(-1.0, 1.0, N)
pdf_spin = np.ones_like(chi1z)*(1./2)*(1./2)

# Redshift
merger_rate_density = hanabi.hierarchical.merger_rate_density.BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity()
optical_depth = hanabi.lensing.optical_depth.HannukselaEtAl2019OpticalDepth()
p_z = hanabi.hierarchical.p_z.NotLensedSourceRedshiftProbDist(merger_rate_density=merger_rate_density, optical_depth=optical_depth)
zs = p_z.sample(size=N)
pdf_z = p_z.prob(zs)

binaries['mtot'] = (m1_src + m2_src)*(1 + zs)
binaries['q'] = q
binaries['z'] = zs
binaries['chi1x'] = chi1x
binaries['chi1y'] = chi1y
binaries['chi1z'] = chi1z
binaries['chi2x'] = chi2x
binaries['chi2y'] = chi2y
binaries['chi2z'] = chi2z

# Calculating the BBH selection function alpha
model = pdetclassifier.loadnetwork(trained_model)
predictions = pdetclassifier.predictnetwork(model, binaries)
integral = np.sum(predictions).astype(float)/N # Monte-Carlo integration

# Writing the results to a HDF5 file
output = {"m1_src": m1_src, "m2_src": m2_src, "sampling_pdf": pdf_mass+pdf_spin+pdf_z, "predictions": predictions}
output.update(binaries)
del output["N"]
output_keys = list(output.keys())
output_dt = np.dtype({"names": output_keys, "formats": [float]*len(output_keys)})
output = np.rec.array(list(output.values()), dtype=output_dt)
with h5py.File(label+".h5", "w") as f:
	dataset = f.create_dataset("binaries", data=output)
	dataset.attrs["N"] = N
	dataset.attrs["alpha"] = integral

# Calculating the lensing selection function beta