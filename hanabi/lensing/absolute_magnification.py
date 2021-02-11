import numpy as np
from bilby.core.prior import *

class SISPowerLawAbsoluteMagnification(PowerLaw):
    def __init__(self, name=None, latex_label=None, unit=None, boundary=None):
        super(SISPowerLawAbsoluteMagnification, self).__init__(
            alpha=-3,
            minimum=2,
            maximum=np.inf,
            name=name,
            unit=unit,
            latex_label=latex_label,
            boundary=boundary
        )
