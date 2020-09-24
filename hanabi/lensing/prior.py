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
        return int(np.floor(self.N*val) + self.minimum)
    
    def prob(self, val):
        return ((val >= self.minimum) & (val < self.maximum))/float(self.N)

class RelativeMagnificationPoorMan(bilby.core.prior.Prior):
    def __init__(self):
        """
        RelativeMagnificationPoorMan
        ----------

        Implement the poor man's prior for relative magnification
        """
        pass

