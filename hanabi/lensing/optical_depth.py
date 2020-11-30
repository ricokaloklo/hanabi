# optical_depth.py
# evluate p(lensed|z_src)
# evaluate p(z_src|lensed) \propto p(lensed|z_src) * p(z_src)/p(lensed)
import numpy as np
import scipy.integrate
import astropy.cosmology
from astropy.cosmology import Planck15
from astropy import units as u

class OpticalDepth(object):
    def __init__(self):
        pass

    def evaluate(self, z):
        return NotImplementedError

class OpticalDepthFromFile(object):
    pass

class TurnerEtAlOpticalDepth(OpticalDepth):
    """
    Analytical model from Turner, Ostriker & Gott 1984, Fukugita & Turner 1991
    """
    def __init__(self, norm, cosmo=Planck15):
        self.norm = norm
        self.cosmology = cosmo

    def evaluate(self, z):
        D_c = self.cosmology.comoving_transverse_distance(z).to(u.Gpc)
        return norm*(D_c)**3

class NgEtAlOpticalDepth(TurnerEtAlOpticalDepth):
    def __init__(self):
        super(NgEtAlOpticalDepth, self).__init__(
            4.17e-6/6.3,
            cosmo=astropy.cosmology.FlatLambdaCDM(
                70,
                1-0.7
            )
        )

class HarisEtAlOpticalDepth(TurnerEtAlOpticalDepth):
    def __init__(self):
        super(HarisEtAlOpticalDepth, self).__init__(
            4.17e-6,
            cosmo=astropy.cosmology.FlatLambdaCDM(
                70,
                1-0.7
            )
        )

