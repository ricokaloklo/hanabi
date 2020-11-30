import numpy as np
from bilby.core.prior import PowerLaw

class AbsoluteMagnificationProbDist(object):
    def __init__(self):
        pass

    def prob(self, mu_abs):
        return NotImplementedError

    def ln_prob(self, mu_abs):
        return np.log(self.prob(mu_abs))

class PowerLawAbsoluteMagnificationProbDist(AbsoluteMagnificationProbDist):
    def __init__(self, mu_abs_min=2):
        self._PowerLaw = PowerLaw(-3, mu_abs_min, np.inf)

    def prob(self, mu_abs):
        return self._PowerLaw.prob(mu_abs)
