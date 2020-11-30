# optical_depth.py
# evluate p(lensed|z_src)
# evaluate p(z_src|lensed) \propto p(lensed|z_src) * p(z_src)/p(lensed)
import numpy as np
import astropy.cosmology
from astropy.cosmology import Planck15

class OpticalDepth(object):
    def __init__(self):
        pass

    def evaluate(self, z):
        return NotImplementedError

class OpticalDepthFromFile(object):
    pass

class TurnerEtAl1984OpticalDepth(OpticalDepth):
    """
    Analytical model from Turner, Ostriker & Gott 1984
    """
    def __init__(self, norm, cosmo=Planck15):
        self.norm = norm
        self.cosmology = cosmo

    def evaluate(self, z):
        d_C = self.cosmology.comoving_distance(z).to('Gpc').value
        return self.norm*(d_C)**3

class HannukselaEtAl2019OpticalDepth(TurnerEtAl1984OpticalDepth):
    def __init__(self):
        super(HannukselaEtAl2019OpticalDepth, self).__init__(
            0.0017/(Planck15.hubble_distance.to('Gpc').value**3),
            cosmo=Planck15
        )

class HarisEtAl2018OpticalDepth(TurnerEtAl1984OpticalDepth):
    def __init__(self):
        cosmo=astropy.cosmology.FlatLambdaCDM(
            70,
            1-0.7
        )
        super(HarisEtAl2018OpticalDepth, self).__init__(
            4.17e-6,
            cosmo=cosmo
        )
