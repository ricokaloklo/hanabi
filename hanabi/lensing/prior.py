import numpy as np
import bilby.core.prior

class DiscreteUniform(bilby.core.prior.Prior):
    def __init__(self, name=None, latex_label=None, unit=None, minimum=1, N=-1):
        """
        DiscreteUniform
        ----------

        Sample uniformly over a discrete set of integers starting from minimum,...,N+minimum-1
        Generate samples using inverse CDF method
        """
        # Check the validity of the input N
        if isinstance(N, int) and N >= 0:
            self.N = N
        else:
            raise ValueError("The parameter N must be provided with a positive integer")

        super(DiscreteUniform, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=int(minimum),
            maximum=int((self.N + minimum) - 1),
            boundary=None
        )

    def rescale(self, val):
        return np.floor(self.N*val) + self.minimum
    
    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum))/float(self.N) * (np.modf(val)[0] == 0).astype(int)

    def cdf(self, val):
        return (val <= self.maximum) * (np.floor(val) - self.minimum + 1)/float(self.N) + (val > self.maximum)

class RelativeMagnificationPoorMan(bilby.core.prior.Prior):
    def __init__(self, name=None, latex_label=None, unit=None, maximum=np.inf):
        """
        RelativeMagnificationPoorMan
        ----------

        Implement the poor man's prior for relative magnification \mu_{rel}:
        p(\mu_{rel}) = \frac{2 \mu_{rel, max}^2}{2 \mu_{rel, max}^2 - 1} * 
        \begin{cases}
        \mu_{rel} & \text{if } 0 < \mu_{rel} <= 1, \\
        \mu_{rel}^{-3} & \text{if } 1 < \mu_{rel} <= \mu_{rel, max}, \\
        0 & \text{otherwise}
        \end{cases}
        """
        self.norm = 1.0/(1.0 - 1.0/(2.0 * np.power(maximum, 2)))
        super(RelativeMagnificationPoorMan, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=0.,
            maximum=maximum,
            boundary=None
        )

    def rescale(self, val):
        # Using generic inverse CDF method
        return ((0 <= val) & (val < 0.5)) * np.sqrt(val/(self.norm*0.5)) + ((0.5 <= val) & (val < 1)) * np.sqrt(0.5/(1.0 - val/self.norm))

    def prob(self, val):
        return self.norm * (((0 < val) & (val <= 1)) * val + ((1 < val) & (val <= self.maximum)) * np.nan_to_num(1.0/np.power(val, 3)))

    def cdf(self, val):
        return self.norm * (((0 < val) & (val <= 1)) * 0.5*np.power(val, 2) + ((1 < val) & (val <= self.maximum)) * (1 - np.nan_to_num(0.5/np.power(val, 2)))) + (val > self.maximum)

