import os
import numpy as np
import h5py
import pandas as pd
import h5py
import copy
import tqdm
import logging
import scipy.interpolate
import bilby
from schwimmbad import SerialPool, MultiPool
from .source_population_model import Marginalized
from .merger_rate_density import MarginalizedMergerRateDensity
from .p_z import LensedSourceRedshiftProbDist, NotLensedSourceRedshiftProbDist
from .marginalized_likelihood import DetectorFrameComponentMassesFromSourceFrame, LuminosityDistancePriorFromAbsoluteMagnificationRedshift
from .utils import setup_logger, enforce_mass_ordering, write_to_hdf5
from .cupy_utils import _GPU_ENABLED
from ..lensing.absolute_magnification import SISPowerLawAbsoluteMagnification
from ..lensing.optical_depth import NotLensedOpticalDepth, Oguri2019StrongLensingProb

__prog__ = "selection_function.py"

def compute_mean_selection_function(selection_function, N_avg, pool=None):
    if pool is None:
        pool = SerialPool()
    elif isinstance(pool, int):
        pool = MultiPool(pool)
    elif isinstance(pool, (SerialPool, MultiPool)):
        pool = pool
    else:
        raise TypeError("Does not understand the given multiprocessing pool.")

    out = list(pool.starmap(selection_function.evaluate, [() for _ in range(N_avg)]))
    avg = np.average(out)
    pool.close()
    return avg

class SelectionFunction(object):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity()
    ):
        self.mass_src_pop_model = mass_src_pop_model
        self.spin_src_pop_model = spin_src_pop_model
        self.merger_rate_density_src_pop_model = merger_rate_density_src_pop_model
    
    def expected_number_of_mergers(self, T_obs, IFAR_threshold=1):
        return 1.0

    def evaluate(self, IFAR_threshold=1):
        # T_obs does not matter in evaluating the selection function
        N_tot = self.merger_rate_density_src_pop_model.total_number_of_mergers(1.0)
        N_exp = self.expected_number_of_mergers(1.0, IFAR_threshold=IFAR_threshold)

        return N_exp/N_tot

class LensedBinaryBlackHoleSelectionFunctionFromMachineLearning(SelectionFunction):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity(),
        abs_magn_dist=[SISPowerLawAbsoluteMagnification(), SISPowerLawAbsoluteMagnification()],
        optical_depth=Oguri2019StrongLensingProb(),
        filename=None,
        trained_model=None,
        seed=None,
        N_inj=int(1e7),
        N_z=100,
        N_img=2,
    ):
        super(LensedBinaryBlackHoleSelectionFunctionFromMachineLearning, self).__init__(
            mass_src_pop_model=mass_src_pop_model,
            spin_src_pop_model=spin_src_pop_model,
            merger_rate_density_src_pop_model=merger_rate_density_src_pop_model
        )
        self.abs_magn_dist = abs_magn_dist
        self.optical_depth = optical_depth

        if seed is not None:
            np.random.seed(seed)
        self.filename = filename
        self.trained_model = trained_model
        self.N_inj = N_inj
        self.N_z = N_z
        self.N_img = N_img

        if len(self.abs_magn_dist) != self.N_img:
            raise ValueError("Number of absolute mangification distributions given does not match the number of images")

        logger = logging.getLogger(__prog__)
        logger.info("Building a fast interpolant from luminosity distance to redshift")
        # Building a fast interpolant for dL -> redshift
        z_int = np.linspace(0, self.merger_rate_density_src_pop_model.population_parameter_dict["redshift_max"], num=100)
        dL_int = bilby.gw.conversion.redshift_to_luminosity_distance(z_int)
        spline_int = scipy.interpolate.splrep(dL_int, z_int, s=0)
        self.z_from_dL_interpolant = lambda dL: scipy.interpolate.splev(dL, spline_int, der=0)

        if self.trained_model is not None:
            if self.filename is None:
                self.filename = "{}_{}".format("fiducial_beta", os.path.basename(self.trained_model))
            self.make_fiducial_predictions()

    def make_fiducial_predictions(self):
        # NOTE This requires tensorflow to be installed
        from hanabi.hierarchical.pdetclassifier import pdetclassifier
        logger = logging.getLogger(__prog__)

        zmin = 0
        zmax = self.merger_rate_density_src_pop_model.population_parameter_dict["redshift_max"]
        mmin = self.mass_src_pop_model.population_parameter_dict.get("mmin", 5)
        mmax = self.mass_src_pop_model.population_parameter_dict.get("mmax", 100)
        mmin_det = mmin*(1+zmin) # redshifted
        mmax_det = mmax*(1+zmax) # redshifted

        # Fiduical detector-frame masses
        fiducial_detector_frame_mass_model = {
            "mass_1": bilby.core.prior.Uniform(minimum=mmin_det, maximum=mmax_det),
            "mass_2": bilby.core.prior.Uniform(minimum=mmin_det, maximum=mmax_det),
        }
        # Fiducial spins
        fiducial_intrinsic_pop_models = {
            "a_1": bilby.core.prior.Uniform(minimum=0, maximum=1),
            "a_2": bilby.core.prior.Uniform(minimum=0, maximum=1),
            "tilt_1": bilby.core.prior.Sine(),
            "tilt_2": bilby.core.prior.Sine(),
            "phi_1": bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi),
            "phi_2": bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi),
        }

        m1 = fiducial_detector_frame_mass_model["mass_1"].sample(size=self.N_inj)
        m2 = fiducial_detector_frame_mass_model["mass_2"].sample(size=self.N_inj)
        m1, m2 = enforce_mass_ordering(m1, m2)
        q = m2/m1
        pdf_mass_fiducial = fiducial_detector_frame_mass_model["mass_1"].prob(m1)*fiducial_detector_frame_mass_model["mass_2"].prob(m2)

        a_1, tilt_1, phi_1 = [fiducial_intrinsic_pop_models[k].sample(size=self.N_inj) for k in ["a_1", "tilt_1", "phi_1"]]
        a_2, tilt_2, phi_2 = [fiducial_intrinsic_pop_models[k].sample(size=self.N_inj) for k in ["a_2", "tilt_2", "phi_2"]]
        spin_1x, spin_1y, spin_1z = pdetclassifier.sperical_to_cartesian(a_1, tilt_1, phi_1)
        spin_2x, spin_2y, spin_2z = pdetclassifier.sperical_to_cartesian(a_2, tilt_2, phi_2)
        pdf_spin_fiducial = fiducial_intrinsic_pop_models["a_1"].prob(a_1) * \
                            fiducial_intrinsic_pop_models["a_2"].prob(a_2) * \
                            fiducial_intrinsic_pop_models["tilt_1"].prob(tilt_1) * \
                            fiducial_intrinsic_pop_models["tilt_2"].prob(tilt_2) * \
                            fiducial_intrinsic_pop_models["phi_1"].prob(phi_1) * \
                            fiducial_intrinsic_pop_models["phi_2"].prob(phi_2)        

        # Fiduical apparent luminosity distance distribution
        fiducial_apparent_luminosity_distance_dist = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=zmin, maximum=bilby.gw.conversion.redshift_to_luminosity_distance(zmax), unit='Mpc')
        apparent_dLs = [fiducial_apparent_luminosity_distance_dist.sample(size=self.N_inj) for img in range(self.N_img)]
        apparent_zs = [self.z_from_dL_interpolant(apparent_dL) for apparent_dL in apparent_dLs]
        pdf_dLs_fiducial = []
        for apparent_dL in apparent_dLs:
            pdf_dLs_fiducial.append(fiducial_apparent_luminosity_distance_dist.prob(apparent_dL))

        binaries = {}
        binaries = pdetclassifier.generate_binaries(self.N_inj)
        binaries['mtot'] = m1 + m2
        binaries['q'] = q
        binaries['chi1x'] = spin_1x
        binaries['chi1y'] = spin_1y
        binaries['chi1z'] = spin_1z
        binaries['chi2x'] = spin_2x
        binaries['chi2y'] = spin_2y
        binaries['chi2z'] = spin_2z

        logger = logging.getLogger(__prog__)
        logger.info("Generating fiducial predictions from pdetclassifer")
        model = pdetclassifier.loadnetwork(self.trained_model)
        images = []
        for img in range(self.N_img):
            images.append(copy.deepcopy(binaries))
            images[img]['z'] = apparent_zs[img]
            images[img]['predictions'] = pdetclassifier.predictnetwork(model, images[img])

        # Save output to file
        logger.info(f"Saving to file {self.filename}")
        output_dataset = \
        {
            "binaries": {
                "mass_1": m1,
                "mass_2": m2,
                "spin_1x": spin_1x,
                "spin_1y": spin_1y,
                "spin_1z": spin_1z,
                "spin_2x": spin_2x,
                "spin_2y": spin_2y,
                "spin_2z": spin_2z,
                "pdf_mass": pdf_mass_fiducial,
                "pdf_spin": pdf_spin_fiducial,
            }
        }
        for img in range(self.N_img):
            output_dataset["apparent_luminosity_distance_{}".format(img+1)] = {"d_L": apparent_dLs[img], "pdf": pdf_dLs_fiducial[img]}
            output_dataset["predictions_{}".format(img+1)] = {"prediction": images[img]['predictions']}

        write_to_hdf5(
            self.filename,
            output_dataset,
            {
                "mmin_det": mmin_det,
                "mmax_det": mmax_det,
                "N_inj": self.N_inj,
                "N_img": self.N_img,
            }
        )

    def load_from_file(self):
        logger = logging.getLogger(__prog__)
        logger.info(f"Reading from file {self.filename}")
        self.f = h5py.File(self.filename, "r")
        self.fiducial_binaries = self.f["binaries"]
        self.pdf_mass_fiducial = self.f["binaries"]["pdf_mass"]
        self.pdf_spin_fiducial = self.f["binaries"]["pdf_spin"]
        self.apparent_dLs = []
        self.pdf_dLs_fiducial = []
        self.predictions = []
        for img in range(self.N_img):
            self.apparent_dLs.append(self.f["apparent_luminosity_distance_{}".format(img+1)]["d_L"])
            self.pdf_dLs_fiducial.append(self.f["apparent_luminosity_distance_{}".format(img+1)]["pdf"])
            self.predictions.append(self.f["predictions_{}".format(img+1)]["prediction"])

    def evaluate(self):
        self.load_from_file()
        m1 = self.fiducial_binaries["mass_1"]
        m2 = self.fiducial_binaries["mass_2"]
        # Note that spins are redshift independent
        spin_1x, spin_1y, spin_1z = [self.fiducial_binaries[k] for k in ["spin_1x", "spin_1y", "spin_1z"]]
        spin_2x, spin_2y, spin_2z = [self.fiducial_binaries[k] for k in ["spin_2x", "spin_2y", "spin_2z"]]

        if _GPU_ENABLED:
            import cupy as xp
        else:
            import numpy as xp
        m1 = xp.asarray(m1)
        m2 = xp.asarray(m2)
        spin_1x = xp.asarray(spin_1x)
        spin_1y = xp.asarray(spin_1y)
        spin_1z = xp.asarray(spin_1z)
        spin_2x = xp.asarray(spin_2x)
        spin_2y = xp.asarray(spin_2y)
        spin_2z = xp.asarray(spin_2z)
        pdf_spin_fiducial = xp.asarray(self.pdf_spin_fiducial)
        pdf_mass_fiducial = xp.asarray(self.pdf_mass_fiducial)

        pdf_spin_pop = self.spin_src_pop_model.prob({
            "spin_1x": spin_1x,
            "spin_1y": spin_1y,
            "spin_1z": spin_1z,
            "spin_2x": spin_2x,
            "spin_2y": spin_2y,
            "spin_2z": spin_2z,
        })
        weights_spin = pdf_spin_pop/pdf_spin_fiducial

        for img in range(self.N_img):
            self.predictions[img] = xp.asarray(self.predictions[img])
            self.pdf_dLs_fiducial[img] = xp.asarray(self.pdf_dLs_fiducial[img])
            self.apparent_dLs[img] = xp.asarray(self.apparent_dLs[img])

        def epsilon(z_src):
            det_mass_pop_dist = DetectorFrameComponentMassesFromSourceFrame(self.mass_src_pop_model, z_src)
            pdf_mass_pop = det_mass_pop_dist.prob({"mass_1": m1, "mass_2": m2})
            weights_mass = pdf_mass_pop/pdf_mass_fiducial
            weights_source = weights_mass * weights_spin
            integrand = weights_source

            for img in range(self.N_img):
                pdf_dL_fiducial = self.pdf_dLs_fiducial[img]
                dL_pop_dist = LuminosityDistancePriorFromAbsoluteMagnificationRedshift(self.abs_magn_dist[img], z_src)
                pdf_dL_pop = dL_pop_dist.prob(self.apparent_dLs[img])
                weights_dL = pdf_dL_pop/pdf_dL_fiducial
                integrand *= self.predictions[img]*weights_dL

            return float(xp.sum(integrand)/float(self.N_inj))

        logger = logging.getLogger(__prog__)
        logger.info("Integrating over source redshift")
        z_dist = LensedSourceRedshiftProbDist(merger_rate_density=self.merger_rate_density_src_pop_model, optical_depth=self.optical_depth)
        zs = z_dist.sample(size=self.N_z)
        epsilons = []
        for z in tqdm.tqdm(zs):
            epsilons.append(epsilon(z))

        beta = np.sum(epsilons).astype(float)/self.N_z

        self.f.close()
        return beta

class BinaryBlackHoleSelectionFunctionFromMachineLearning(SelectionFunction):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity(),
        optical_depth=NotLensedOpticalDepth(),
        filename=None,
        trained_model=None,
        seed=None,
        N_inj=int(1e7),
    ):
        super(BinaryBlackHoleSelectionFunctionFromMachineLearning, self).__init__(
            mass_src_pop_model=mass_src_pop_model,
            spin_src_pop_model=spin_src_pop_model,
            merger_rate_density_src_pop_model=merger_rate_density_src_pop_model
        )
        self.optical_depth = optical_depth

        if seed is not None:
            np.random.seed(seed)
        self.filename = filename
        self.trained_model = trained_model
        self.N_inj = N_inj

        if self.trained_model is not None:
            if self.filename is None:
                self.filename = "{}_{}".format("fiducial_alpha", os.path.basename(self.trained_model))
            self.make_fiducial_predictions()

    def make_fiducial_predictions(self):
        # NOTE This requires tensorflow to be installed
        from hanabi.hierarchical.pdetclassifier import pdetclassifier

        zmin = 0
        zmax = self.merger_rate_density_src_pop_model.population_parameter_dict.get("redshift_max", 10)
        mmin = self.mass_src_pop_model.population_parameter_dict.get("mmin", 5)
        mmax = self.mass_src_pop_model.population_parameter_dict.get("mmax", 100)
        # Fiducial intrinsic population model. We generate samples from these distribution instead
        fiducial_intrinsic_pop_models = {
            "m1_src": bilby.core.prior.Uniform(minimum=mmin, maximum=mmax),
            "m2_src": bilby.core.prior.Uniform(minimum=mmin, maximum=mmax),
            "a_1": bilby.core.prior.Uniform(minimum=0, maximum=1),
            "a_2": bilby.core.prior.Uniform(minimum=0, maximum=1),
            "tilt_1": bilby.core.prior.Sine(),
            "tilt_2": bilby.core.prior.Sine(),
            "phi_1": bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi),
            "phi_2": bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi),
        }
        # Fiducial redshift model. We generate samples from this distribution instead
        fiducial_redshift_pop_model = bilby.gw.prior.UniformComovingVolume(name='redshift', minimum=zmin, maximum=zmax)

        # Setting up a fiducial population
        binaries = pdetclassifier.generate_binaries(self.N_inj)
        # Intrinsic parameters
        m1_src = fiducial_intrinsic_pop_models["m1_src"].sample(size=self.N_inj)
        m2_src = fiducial_intrinsic_pop_models["m2_src"].sample(size=self.N_inj)
        m1_src, m2_src = enforce_mass_ordering(m1_src, m2_src)
        q = m2_src/m1_src
        pdf_mass_fiducial = np.ones_like(m1_src)*(fiducial_intrinsic_pop_models["m1_src"].prob(m1_src))*(fiducial_intrinsic_pop_models["m2_src"].prob(m2_src))
        a_1, tilt_1, phi_1 = [fiducial_intrinsic_pop_models[k].sample(size=self.N_inj) for k in ["a_1", "tilt_1", "phi_1"]]
        a_2, tilt_2, phi_2 = [fiducial_intrinsic_pop_models[k].sample(size=self.N_inj) for k in ["a_2", "tilt_2", "phi_2"]]
        spin_1x, spin_1y, spin_1z = pdetclassifier.sperical_to_cartesian(a_1, tilt_1, phi_1)
        spin_2x, spin_2y, spin_2z = pdetclassifier.sperical_to_cartesian(a_2, tilt_2, phi_2)
        pdf_spin_fiducial = fiducial_intrinsic_pop_models["a_1"].prob(a_1) * \
                            fiducial_intrinsic_pop_models["a_2"].prob(a_2) * \
                            fiducial_intrinsic_pop_models["tilt_1"].prob(tilt_1) * \
                            fiducial_intrinsic_pop_models["tilt_2"].prob(tilt_2) * \
                            fiducial_intrinsic_pop_models["phi_1"].prob(phi_1) * \
                            fiducial_intrinsic_pop_models["phi_2"].prob(phi_2)

        # Redshift
        z = fiducial_redshift_pop_model.sample(size=self.N_inj)
        pdf_z_fiducial = fiducial_redshift_pop_model.prob(z)

        binaries['mtot'] = (m1_src + m2_src)*(1 + z)
        binaries['q'] = q
        binaries['chi1x'] = spin_1x
        binaries['chi1y'] = spin_1y
        binaries['chi1z'] = spin_1z
        binaries['chi2x'] = spin_2x
        binaries['chi2y'] = spin_2y
        binaries['chi2z'] = spin_2z
        binaries['z'] = z

        logger = logging.getLogger(__prog__)
        logger.info("Generating fiducial predictions from pdetclassifer")
        model = pdetclassifier.loadnetwork(self.trained_model)
        predictions = pdetclassifier.predictnetwork(model, binaries)

        # Save output to file
        logger.info(f"Saving to file {self.filename}")
        write_to_hdf5(
            self.filename,
            {
                "binaries": {
                    "mass_1_source": m1_src,
                    "mass_2_source": m2_src,
                    "spin_1x": spin_1x,
                    "spin_1y": spin_1y,
                    "spin_1z": spin_1z,
                    "spin_2x": spin_2x,
                    "spin_2y": spin_2y,
                    "spin_2z": spin_2z,
                    "pdf_mass": pdf_mass_fiducial,
                    "pdf_spin": pdf_spin_fiducial,
                },
                "redshifts": {"z": z, "pdf": pdf_z_fiducial},
                "predictions": {"prediction": predictions},
            },
            {
                "mmin": mmin,
                "mmax": mmax,
                "zmin": zmin,
                "zmax": zmax,
                "N_inj": self.N_inj,
            }
        )

    def load_from_file(self):
        logger = logging.getLogger(__prog__)
        logger.info(f"Reading from file {self.filename}")
        self.f = h5py.File(self.filename, "r")
        self.fiducial_binaries = self.f["binaries"]
        self.pdf_mass_fiducial = self.f["binaries"]["pdf_mass"]
        self.pdf_spin_fiducial = self.f["binaries"]["pdf_spin"]
        self.fiducial_z = self.f["redshifts"]["z"]
        self.pdf_z_fiducial = self.f["redshifts"]["pdf"]
        self.predictions = self.f["predictions"]["prediction"]

    def evaluate(self):
        self.load_from_file()

        # Perform MC reweighting
        if _GPU_ENABLED:
            import cupy as xp
        else:
            import numpy as xp

        m1_src = self.fiducial_binaries["mass_1_source"]
        m2_src = self.fiducial_binaries["mass_2_source"]
        spin_1x, spin_1y, spin_1z = [self.fiducial_binaries[k] for k in ["spin_1x", "spin_1y", "spin_1z"]]
        spin_2x, spin_2y, spin_2z = [self.fiducial_binaries[k] for k in ["spin_2x", "spin_2y", "spin_2z"]]
        pdf_mass_fiducial = self.pdf_mass_fiducial
        pdf_spin_fiducial = self.pdf_spin_fiducial

        # Move data to GPU if needed
        m1_src = xp.asarray(m1_src)
        m2_src = xp.asarray(m2_src)
        spin_1x = xp.asarray(spin_1x)
        spin_1y = xp.asarray(spin_1y)
        spin_1z = xp.asarray(spin_1z)
        spin_2x = xp.asarray(spin_2x)
        spin_2y = xp.asarray(spin_2y)
        spin_2z = xp.asarray(spin_2z)
        pdf_mass_fiducial = xp.asarray(pdf_mass_fiducial)
        pdf_spin_fiducial = xp.asarray(pdf_spin_fiducial)

        pdf_mass_pop = self.mass_src_pop_model.prob({"mass_1_source": m1_src, "mass_2_source": m2_src}, axis=0)
        weights_mass = pdf_mass_pop/pdf_mass_fiducial      
        pdf_spin_pop = self.spin_src_pop_model.prob({
            "spin_1x": spin_1x,
            "spin_1y": spin_1y,
            "spin_1z": spin_1z,
            "spin_2x": spin_2x,
            "spin_2y": spin_2y,
            "spin_2z": spin_2z,
        })
        weights_spin = pdf_spin_pop/pdf_spin_fiducial

        weights_source = weights_mass * weights_spin

        z = xp.asarray(self.fiducial_z)
        pz = NotLensedSourceRedshiftProbDist(merger_rate_density=self.merger_rate_density_src_pop_model, optical_depth=self.optical_depth)
        pdf_z_fiducial = xp.asarray(self.pdf_z_fiducial)
        pdf_z_pop = pz.prob(z)
        weights_z = pdf_z_pop/pdf_z_fiducial

        predictions = xp.asarray(self.predictions)
        alpha = xp.sum(predictions*weights_source*weights_z).astype(float)/(len(m1_src))

        self.f.close()
        # NOTE If using numpy, alpha is a scalar but if using cupy, alpha is a 0-d array
        return float(alpha)

class BinaryBlackHoleSelectionFunctionFromInjection(SelectionFunction):
    def __init__(
        self,
        mass_src_pop_model=Marginalized(),
        spin_src_pop_model=Marginalized(),
        merger_rate_density_src_pop_model=MarginalizedMergerRateDensity(),
        filename=None
    ):
        super(BinaryBlackHoleSelectionFunctionFromInjection, self).__init__(
            mass_src_pop_model=mass_src_pop_model,
            spin_src_pop_model=spin_src_pop_model,
            merger_rate_density_src_pop_model=merger_rate_density_src_pop_model
        )

        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "data", "o3a_bbhpop_inj_info.hdf")
        pop_inj_info = h5py.File(filename, "r")
        # Read in dataset and attributes from the file
        for key, value in pop_inj_info.attrs.items():
            if key == "N_exp/R(z=0)":
                # This particular key does not play well
                key = "N_exp_over_R_0"
            setattr(self, "pop_inj_{}".format(key), value)

        column_names = list(pop_inj_info["injections"])
        self.pop_inj_info = pd.DataFrame(data={column: pop_inj_info["injections"][column] for column in column_names})

        pop_inj_info.close()

        self.pop_inj_analysis_time_yr = self.pop_inj_analysis_time_s/(365.25*24*3600)
        # Rename the column
        self.pop_inj_info.rename(columns={
            "mass1_source": "mass_1_source",
            "mass2_source": "mass_2_source",
            "spin1z": "spin_1z",
            "spin2z": "spin_2z",
        }, inplace=True)

    def expected_number_of_mergers(self, T_obs, IFAR_threshold=1):
        # Default IFAR threshold= 1 yr
        # Detected either by gstlal, pycbc_full or pycbc_bbh
        detected = (self.pop_inj_info["ifar_gstlal"] > IFAR_threshold) | (self.pop_inj_info["ifar_pycbc_full"] > IFAR_threshold) | (self.pop_inj_info["ifar_pycbc_bbh"] > IFAR_threshold)

        # Actually compute the log of expected number
        log_dN = np.where(
            detected,
            self.mass_src_pop_model.ln_prob(self.pop_inj_info[["mass_1_source", "mass_2_source"]], axis=0) + \
            self.spin_src_pop_model.ln_prob(self.pop_inj_info[["spin_1z", "spin_2z"]], axis=0) + \
            self.merger_rate_density_src_pop_model.ln_dN_over_dz(self.pop_inj_info),
            np.NINF
        )
        log_p_draw = np.log(self.pop_inj_info["sampling_pdf"].to_numpy())
        log_N_exp = np.log(T_obs) + np.logaddexp.reduce(log_dN - log_p_draw) - np.log(self.pop_inj_total_generated)

        return np.exp(log_N_exp)

