import numpy as np
import bilby.gw.conversion
import pandas as pd
import scipy.integrate
from .source_population_model import SourcePopulationModel, Marginalized
from astropy.cosmology import Planck15

class MergerRateDensity(SourcePopulationModel):
    """Base class for a merger rate density model.

    The merger rate density model :math: `\mathcal{R}(z)` is 
    the number of mergers per unit co-moving volume per unit time
    in the source frame.
    
    The unit here is :math:`{\rm Gpc}^{-3} T^{-1}`, where the
    unit of time is arbitrary.

    """
    def __init__(self, population_parameter_dict):
        """Initialize a `MergerRateDensity` object.

        This __init__ method also computes the normalization constant
        needed for later use.

        Parameters
        ----------
        population_parameter_dict : dict
            A dictionary of population parameters
            for the merger rate density model. The key
            `redshift_max` must be set.

        Notes
        -----
        TODO: allow cosmology other than Planck15.

        """
        super(MergerRateDensity, self).__init__(
            signal_parameter_names=["redshift"],
            population_parameter_dict=population_parameter_dict
        )
        self.cosmology = Planck15
        assert "redshift_max" in population_parameter_dict.keys(), "redshift_max must be given"
        self.normalization = 1.0
        self.normalization = self.compute_normalization()

    def evaluate(self, z):
        """Return the numerical value for R(z).

        This function should be overwritten by each subclass.

        Parameters
        ----------
        z : array_like
            Redshifts to evaluate.

        """
        raise NotImplementedError

    def _prob(self, dataset):
        """Compute the probability density.

        Note that the function `compute_normalization` must be
        called a priori for the returned value be normalized.
        This internal function should be called with caution.

        This function converts the merger rate density :math:`\mathcal{R}(z)`
        into a probability density :math:`p(z)`, which is given by

        ..math:: p(z) \propto \frac{1}{z} \frac{dV_{\rm c}}{dz} \mathcal{R}(z)

        Parameters
        ----------
        dataset : dict or pandas DataFrame
            Input dataset, at least one of the following
            key/column should be present:
            `redshift`, `luminosity_distance`, or `comoving_distance`.

        Returns
        -------
        array_like
            The probability density.

        """
        # NOTE astropy.cosmology, by default, uses Mpc as the unit
        dVc_dz = 4.0*np.pi*self.cosmology.differential_comoving_volume(dataset["redshift"]).to('Gpc^3/sr').value
        return (1.0/self.normalization) * (1.0/(1.0 + dataset["redshift"])) * self.evaluate(dataset["redshift"]) * dVc_dz

    def compute_normalization(self):
        """Integrate the unnormalized probability density upto `redshift_max`.

        Returns
        -------
        float
            The normalization constant for the probability density.

        Notes
        -----
        TODO: allow adjustable step size (now fixed to 1e-3).

        """
        zs = np.arange(0, self.population_parameter_dict["redshift_max"], step=1e-3)
        out = scipy.integrate.simps(self._prob({'redshift': zs}), zs)
        return out

    def _parameter_conversion(self, dataset):
        if "luminosity_distance" in dataset.keys():
            # Calculate redshift from luminosity distance
            redshift = bilby.gw.conversion.luminosity_distance_to_redshift(
                dataset["luminosity_distance"]
            )
        elif "comoving_distance" in dataset.keys():
            # Calculate redshift from comoving distance
            redshift = bilby.gw.conversion.comoving_distance_to_redshift(
                dataset["comoving_distance"]
            )
        else:
            raise ValueError("No distance measure in dataset")

        dataset["redshift"] = redshift

    def ln_dN_over_dz(self, dataset):
        """Return the natural log of dN/dz.

        Parameters
        ----------
        dataset : dict or pandas DataFrame
            Input dataset, at least one of the following
            key/column should be present:
            `redshift`, `luminosity_distance`, or `comoving_distance`.

        Notes
        -----
        This is *NOT* a normalized probability density.

        """
        return np.log(self._prob(dataset)*self.normalization)

    def total_number_of_mergers(self, T_obs):
        """Compute the total number of mergers within `T_obs`.

        Parameters
        ----------
        T_obs : array_like
            The total observed time. The unit of time is the same as
            the one used when defining the model.

        """
        return T_obs*self.normalization


class MarginalizedMergerRateDensity(Marginalized):
    """Placeholder class when the merger rate density used does not matter.
    """
    def total_number_of_mergers(self, T_obs):
        return 1.0


class PowerLawMergerRateDensity(MergerRateDensity):
    """Wrapper for the `PowerLawRedshift` model implemented in `gwpopulation`.

    The model reads mathematically as

    .. math:: \mathcal{R}(z) = \mathcal{R}_0 (1+z)^{\kappa}

    """
    def __init__(self, R_0, kappa, redshift_max=2.3):
        """Initialize a `PowerLawMergerRateDensity` object.

        Parameters
        ----------
        R_0 : float
            The value of the merger rate density at z = 0.
        kappa: float
            The exponent for the power law.
        redshift_max: float
            The maximum allowed redshift.

        """
        super(PowerLawMergerRateDensity, self).__init__(
            population_parameter_dict={
                'R_0': R_0,
                'kappa': kappa,
                'redshift_max': redshift_max
            }
        )

    def evaluate(self, z):
        """Return the numerical value for R(z).

        Parameters
        ----------
        z : array_like
            Redshifts to evaluate.

        Returns
        -------
        array_like
            The numerical value for R(z).

        """
        return self.population_parameter_dict["R_0"] * \
            np.power(1.+z, self.population_parameter_dict["kappa"]) * \
            ((z >= 0.0) & (z < self.population_parameter_dict["redshift_max"]))

class AnalyticalBBHMergerRateDensity(MergerRateDensity):
    """A generic parametric fit for a merger rate density model.

    The fit is from Eq (13) of arXiv:1807.02584.
    It reads mathematically as

    ..math::  \mathcal{R}(z) = \frac{a_1 e^{a_2 z}}{e^{a_3 z} + a_4}
    when :math:`z` is less than `redshift_max`, and it is zero otherwise.

    """
    def __init__(self, a_1, a_2, a_3, a_4, redshift_max):
        """Initialize a `AnalyticalBBHMergerRateDensity` object

        Parameters
        ----------
        a_1: float
            A parameter for the fit.
        a_2: float
            A parameter for the fit.
        a_3: float
            A parameter for the fit.
        a_4: float
            A parameter for the fit.
        redshift_max: float
            The maximum allowed redshift.
        """
        super(AnalyticalBBHMergerRateDensity, self).__init__(
            population_parameter_dict={
                'a_1': a_1,
                'a_2': a_2,
                'a_3': a_3,
                'a_4': a_4,
                'redshift_max': redshift_max,
            }
        )

        # The merger rate density fit has a unit of Gpc^-3 yr^-1
        self.population_parameter_dict["R_0"] = self.evaluate(0.0)

    def evaluate(self, z):
        """Return the numerical value for R(z).

        Parameters
        ----------
        z : array_like
            Redshifts to evaluate.

        Returns
        -------
        array_like
            The numerical value for R(z).

        """
        a_1 = self.population_parameter_dict['a_1']
        a_2 = self.population_parameter_dict['a_2']
        a_3 = self.population_parameter_dict['a_3']
        a_4 = self.population_parameter_dict['a_4']
        redshift_max = self.population_parameter_dict['redshift_max']

        return np.nan_to_num(a_1*np.exp(a_2*z)/(a_4 + np.exp(a_3*z))*((z >= 0.0) & (z < redshift_max)))

class MadauDickinsonMergerRateDensity(MergerRateDensity):
    """A Madau-Dickson star-formation model-like model.

    The original form is from Eq (15) of 1403.0007, and its
    generalized form used below can be found in arXiv:2003.12152.
    It reads mathematically as

    ..math::  \mathcal{R}(z) = \mathcal{R}_0 \frac{(1+z)^{\alpha}}{1 + (\frac{1+z}{1+z_p})^{\alpha+\beta}}
    when :math:`z` is less than `redshift_max`, and it is zero otherwise.

    """
    def __init__(self, R_0, z_p, alpha, beta, redshift_max):
        """Initialize a `MadauDickinsonMergerRateDensity` object.

        Parameters
        ----------
        R_0 : float
            The value of the merger rate density at z = 0.
        z_p : float
            The peak redshift.
        alpha : float
            The exponent for the initial power-law rise.
        beta : float
            The exponent for the power-law drop after the peak.
        redshift_max: float
            The maximum allowed redshift.

        """
        super(MadauDickinsonMergerRateDensity, self).__init__(
            population_parameter_dict={
                'R_0': R_0,
                'z_p': z_p,
                'alpha': alpha,
                'beta': beta,
                'redshift_max': redshift_max,
            }
        )

    def evaluate(self, z):
        """Return the numerical value for R(z).

        Parameters
        ----------
        z : array_like
            Redshifts to evaluate.

        Returns
        -------
        array_like
            The numerical value for R(z).

        """
        R_0 = self.population_parameter_dict['R_0']
        z_p = self.population_parameter_dict['z_p']
        alpha = self.population_parameter_dict['alpha']
        beta = self.population_parameter_dict['beta']
        redshift_max = self.population_parameter_dict['redshift_max']

        return np.nan_to_num(((R_0*(1.+z)**alpha)/(1.+((1.+z)/(1.+z_p))**(alpha+beta))) *((z >= 0.0) & (z < redshift_max)))

class BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    """A parametric fit for the merger rate density of population I and II stars.
    
    The fit was using data from Belczynski et al. 2017, and the fitting parameters
    are from arXiv:1807.02584.

    """
    def __init__(self):
        super(BelczynskiEtAl2017PopIPopIIStarsBBHMergerRateDensity, self).__init__(
            6.6e3,
            1.6,
            2.1,
            30,
            15
        )

class BelczynskiEtAl2017PopIIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    """A parametric fit for the merger rate density of population III stars.
    
    The fit was using data from Belczynski et al. 2017, and the fitting parameters
    are from arXiv:1807.02584.

    """
    def __init__(self):
        super(BelczynskiEtAl2017PopIIIStarsBBHMergerRateDensity, self).__init__(
            6e4,
            1.0,
            1.4,
            3e6,
            45
        )

class KinugawaEtAl2016PopIIIStarsBBHMergerRateDensity(AnalyticalBBHMergerRateDensity):
    """A parametric fit for the merger rate density of population III stars.
    
    The fit was using data from Kinugawa et al. 2016, and the fitting parameters
    are from arXiv:1807.02584.

    """
    def __init__(self):
        super(KinugawaEtAl2016PopIIIStarsBBHMergerRateDensity, self).__init__(
            1e4,
            0.7,
            1.1,
            500,
            45
        )   