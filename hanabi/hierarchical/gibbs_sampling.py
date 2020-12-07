import numpy as np
import bilby
import pandas as pd
from schwimmbad import SerialPool, MultiPool

class CustomCollapsedBlockedGibbsSampler(object):
    def __init__(self, redshift_result, marginalized_likelihood, random_seed=1234, pool=None:):
        self.redshift_result = redshift_result
        self.marginalized_likelihood = marginalized_likelihood
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.pool = pool

    @staticmethod
    def draw_one_row_from_dataframe(df):
        return df.loc[df.index[np.random.randint(0, len(df)-1)]]

    def draw_from_redshift_posterior(self):
        redshift_samples = self.redshift_result.posterior["redshift"]
        return self.draw_one_row_from_dataframe(redshift_samples)

    def draw_from_inference_posterior_given_redshift(self, z):
        ln_weights = self.marginalized_likelihood.compute_ln_weights_for_component_masses(z) + \
            self.marginalized_likelihood.compute_ln_weights_for_luminosity_distances(z)
        reweighted_samples = bilby.result.rejection_sample(self.marginalized_likelihood.result.posterior, np.exp(ln_weights))
        return self.draw_one_row_from_dataframe(reweighted_samples)

    def draw_one_joint_posterior_sample(self):
        try:
            z_drawn = self.draw_from_redshift_posterior()
            return (z_drawn, draw_from_inference_posterior_given_redshift(z_drawn))
        except:
            # The redshift drawn in this trial leads to no posterior samples after rejection sampling
            return self.draw_joint_posterior_samples()

    def draw_one_joint_posterior_sample_map(self, idx):
        np.random.seed(self.random_seed+idx)
        return self.draw_one_joint_posterior_sample()

    def sample(self, n_samples):
        if self.pool is None:
            pool = SerialPool()
        else:
            if isinstance(self.pool, int):
                pool = MultiPool(self.pool)
            elif isinstance(self.pool, (SerialPool, MultiPool)):
                pool = self.pool
            else:
                raise TypeError("Does not understand the given multiprocessing pool.")

        drawn_samples = list(pool.map(self.draw_one_joint_posterior_sample_map, range(n_samples)))
        pool.close()

        drawn_zs = [drawn_samples[i][0] for i in range(n_samples)]
        drawn_inference_posteriors = [drawn_samples[i][1] for i in range(n_samples)]

        drawn_joint_posterior_samples = pd.DataFrame(drawn_inference_posteriors)
        drawn_joint_posterior_samples["redshift"] = drawn_zs

        return drawn_joint_posterior_samples
