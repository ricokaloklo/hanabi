try:
    import cupy as cp
    _GPU_ENABLED = True
except ImportError:
    _GPU_ENABLED = False

if _GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

import bilby
import numpy as np
import scipy.special
import scipy.integrate
from gwpopulation.cupy_utils import trapz

class PriorDict(bilby.core.prior.PriorDict):
    def ln_prob(self, sample, axis=None):
        if _GPU_ENABLED:
            ln_prob = cp.sum(cp.asarray([self[key].ln_prob(sample[key])
                          for key in sample]), axis=axis)
            return ln_prob
        else:
            return super(PriorDict, self).ln_prob(sample, axis=axis)

# This is a much faster (but stripped down) & CPU/GPU-agnostic implementation of Interped in bilby
class Interped(bilby.core.prior.interpolated.Interped):
    def __init__(self, xx, yy, minimum=xp.nan, maximum=xp.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
        self.xx = xp.asarray(xx)
        self.min_limit = float(xp.amin(self.xx))
        self.max_limit = float(xp.amax(self.xx))
        # In order to use np/cp.interp, we need to make sure that xx is ordered
        sorted_idxs = xp.argsort(self.xx)
        self.xx = self.xx[sorted_idxs]
        self._yy = xp.asarray(yy)[sorted_idxs]
        if self._yy.ndim != 1:
            raise TypeError("yy must be 1D. A {}-D array given.".format(self.yy.dim))
        self.YY = None
        self.probability_density = None
        self.cumulative_distribution = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = lambda x: xp.interp(x, self.xx, self._yy)
        minimum = float(xp.nanmax(xp.array([self.min_limit, minimum])))
        maximum = float(xp.nanmin(xp.array([self.max_limit, maximum])))
        bilby.core.prior.Prior.__init__(self, name=name, latex_label=latex_label, unit=unit, minimum=minimum, maximum=maximum, boundary=boundary)
        self._update_instance()

    def __eq__(self, other):
        other = xp.asarray(other)
        if self.__class__ != other.__class__:
            return False
        if xp.array_equal(self.xx, other.xx) and xp.array_equal(self.yy, other.yy):
            return True
        else:
            return False

    @property
    def yy(self):
        return self._yy

    @yy.setter
    def yy(self, yy):
        self._yy = xp.asarray(yy)
        self.__all_interpolated = lambda x: xp.interp(x, self.xx, self._yy)
        self._update_instance()

    def _update_instance(self):
        self.xx = xp.linspace(self.minimum, self.maximum, len(self.xx)) # This is sorted by def
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        self._yy /= trapz(self._yy, self.xx)
        self.YY = cumtrapz(self._yy, self.xx, initial=0)
        self.YY[-1] = 1
        self.probability_density = lambda x: xp.interp(x, self.xx, self._yy)
        self.cumulative_distribution = lambda x: xp.interp(x, self.xx, self.YY)
        self.inverse_cumulative_distribution = lambda x: xp.interp(x, self.YY, self.xx)

def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    if _GPU_ENABLED:
        y = cp.asarray(y)
        if x is None:
            d = dx
        else:
            x = cp.asarray(x)
            if x.ndim == 1:
                d = cp.diff(x)
                shape = [1]*y.ndim
                shape[axis] = -1
                d = d.reshape(shape)
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
            else:
                d = cp.diff(x, axis=axis)
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

        def tupleset(t, i, value):
            l = list(t)
            l[i] = value
            return tuple(l)
        nd = len(y.shape)
        slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
        slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))

        res = cp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
        if initial is not None:
            if not np.isscalar(initial):
                raise ValueError("`initial` parameter should be a scalar.")

            shape = list(res.shape)
            shape[axis] = 1
            res = cp.concatenate([cp.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

        return res
    else:
        try:
            from scipy.integrate import cumulative_trapezoid as ctz
        except ImportError:
            from scipy.integrate import cumtrapz as ctz
        return ctz(y=y, x=x, dx=dx, axis=axis, initial=initial)

def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    return cumulative_trapezoid(y=y, x=x, dx=dx, axis=axis, initial=initial)

def logsumexp(x):
    if _GPU_ENABLED:
        # NOTE This is a quick-and-dirty implementation
        # FIXME Should contribute to the cupy codebase
        xmax = cp.amax(x)
        t = cp.exp(x - xmax)
        return cp.asnumpy(cp.log(cp.sum(t)) + xmax)
    else:
        return scipy.special.logsumexp(x)