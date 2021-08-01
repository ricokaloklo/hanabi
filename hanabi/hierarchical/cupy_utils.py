try:
    import cupy as cp
    _GPU_ENABLED = True
except ImportError:
    _GPU_ENABLED = False

import bilby
import scipy.special

class PriorDict(bilby.core.prior.PriorDict):
    def ln_prob(self, sample, axis=None):
        if _GPU_ENABLED:
            ln_prob = cp.sum(cp.asarray([self[key].ln_prob(sample[key])
                          for key in sample]), axis=axis)
            return ln_prob
        else:
            return super(PriorDict, self).ln_prob(sample, axis=axis)

def logsumexp(x):
    if _GPU_ENABLED:
        # NOTE This is a quick-and-dirty implementation
        # FIXME Should contribute to the cupy codebase
        xmax = cp.amax(x)
        t = cp.exp(x - xmax)
        return cp.asnumpy(cp.log(cp.sum(t)) + xmax)
    else:
        return scipy.special.logsumexp(x)