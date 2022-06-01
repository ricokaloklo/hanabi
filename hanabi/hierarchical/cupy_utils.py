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
    """CPU/GPU-agonstic `PriorDict` object for `bilby`.

    The stock `PriorDict` is mostly CPU/GPU-agnostic except
    when evaluating the joint probability density, `np.prod`
    and `np.sum` were used explicitly. This object instead
    first checks if GPU is enabled and switchs to using
    `cupy` equivalent when necessary.

    """
    def ln_prob(self, sample, axis=None):
        if _GPU_ENABLED:
            ln_prob = cp.sum(cp.asarray([self[key].ln_prob(sample[key])
                          for key in sample]), axis=axis)
            return ln_prob
        else:
            return super(PriorDict, self).ln_prob(sample, axis=axis)

    def prob(self, sample, **kwargs):
        if _GPU_ENABLED:
            prob = cp.prod(cp.asarray([self[key].prob(sample[key])
                           for key in sample]), **kwargs)
            return prob
        else:
            return super(PriorDict, self).prob(sample, **kwargs)


class Interp(object):
    """CPU/GPU-agonstic interface for `interp`.
    """
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy

    def __call__(self, x):
        return xp.interp(xp.asarray(x), self.xx, self.yy)

class Interped(bilby.core.prior.interpolated.Interped):
    """CPU/GPU-agnostic implementation of `Interped` prior in `bilby`.

    This is a faster but stripped down implementation. Instead of using
    `scipy.interpolate.interp1d`, either `numpy` or `cupy` version of
    the function `interp` is used. The `interp` function assumes that
    the input `xx` is ordered and this is taken care of internally.

    This class is compatibile with `bilby.core.prior.Interped`.

    """
    def __init__(self, xx, yy, minimum=xp.nan, maximum=xp.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Create an interpolated prior function from arrays of xx and yy=p(xx).

        Parameters
        ----------
        xx : array_like
            x values for the to be interpolated prior function.
        yy : array_like
            p(xx) values for the to be interpolated prior function.
        minimum : float
            See superclass.
        maximum : float
            See superclass.
        name : str
            See superclass.
        latex_label : str
            See superclass.
        unit : str
            See superclass.
        boundary : str
            See superclass.

        """
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
        self.__all_interpolated = Interp(self.xx, self._yy)
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
        self.__all_interpolated = Interp(xp.asarray(self.xx), xp.asarray(self._yy))
        self._update_instance()

    def _update_instance(self):
        self.xx = xp.linspace(self.minimum, self.maximum, len(self.xx)) # This is sorted by def
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        self._yy /= trapz(self._yy, self.xx)
        self.YY = cumtrapz(self._yy, self.xx, initial=0)
        self.YY[-1] = 1
        self.probability_density = Interp(self.xx, self._yy)
        self.cumulative_distribution = Interp(self.xx, self.YY)
        self.inverse_cumulative_distribution = Interp(self.YY, self.xx)

def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """CPU/GPU-agnostic implementation of `cumulative_trapezoid`.

    This is using the same algorithm found in
    `scipy.integrate.cumulative_trapezoid`.

    This is a drop-in replacement of the above function in `scipy`,
    and share the same interface.

    See the API reference for `scipy.integrate.cumulative_trapezoid`.

    """
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
    """CPU/GPU-agnostic implementation of `cumtrapz`.

    Note that this is just `cumulative_trapezoid` with a different name.
    """
    return cumulative_trapezoid(y=y, x=x, dx=dx, axis=axis, initial=initial)

def logsumexp(x):
    """CPU/GPU-agnostic implementation of `logsumexp` from `scipy.special`.

    Note that it does not support any of the options for `scipy.special.logsumexp`.

    Parameters
    ----------
    x : array_like
        Input. Could be either a `numpy` or `cupy` array.
    
    Returns
    -------
    array_like
        The value of logsumexp of the input.
    """
    if _GPU_ENABLED:
        # NOTE This is a quick-and-dirty implementation
        # FIXME Should contribute to the cupy codebase
        xmax = cp.amax(x)
        t = cp.exp(x - xmax)
        return cp.asnumpy(cp.log(cp.sum(t)) + xmax)
    else:
        return scipy.special.logsumexp(x)