import numpy as np
import pymc3 as pm
from pymc3.distributions.transforms import ordered
from theano import shared

from likelihood import GRMLike


class GRModel(object):
    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels)
        with pm.Model() as self.model:
            theta = pm.Normal(
                name='theta', shape=(npersons, 1, 1)
            )
            alpha = pm.HalfNormal(
                name='alpha',
                sd=3,
                shape=(1, nitems, 1)
            )

            init_values = np.vstack([np.linspace(-2, 2, nlevels - 1)]*nitems)
            kappa = pm.Normal(
                name='kappa',
                mu=0, sd=3,
                shape=(1, nitems, nlevels - 1),
                transform=ordered,
                testval=init_values[None, ...]
            )

        param_list = [theta, alpha, kappa]
        self.params = {var.name: var for var in param_list}
        self.params['gamma'] = 0
        self.params['sigma'] = 1

        self.likelihood = None
        self.trace = None

    def fit(self, y, draws=500, tune=500, *args, **kwargs):
        self.y = y
        npersons, nitems = y.shape
        nlevels = np.unique(y).size
        if not self.shape == (npersons, nitems, nlevels):
            raise TypeError("y shape does not match model shape")

        if self.likelihood is None:
            self.__data = shared(y, name='y')
            with self.model:
                pi_hat = GRMLike(
                    'pi_hat',
                    **self.params,
                    observed=self.__data
                )
                self.likelihood = pi_hat
        else:
            self.__data.set_value(y)

        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, *args, **kwargs)

        nchains = len(self.trace.chains)
        niters = len(self.trace)
        reps = [nchains, niters]
        for param in self.params:
            if param not in self.trace.varnames:
                entry = {param: np.tile(self.params[param], reps=reps)}
                self.trace.add_values(entry)

        return self.trace

    def summary(self, var_names=None, *args, **kwargs):
        if self.trace is None:
            raise TypeError("Model has not been fit.")
        else:
            return pm.summary(self.trace, var_names, *args, **kwargs)


class TwoPGRM(GRModel):
    pass


class GuessPGRM(GRModel):
    def __init__(self, npersons, nitems, nlevels):
        super(GuessPGRM, self).__init__(npersons, nitems, nlevels)
        with self.model:
            phi = pm.Dirichlet(
                name='phi',
                a=np.ones(nlevels),
                shape=(1, nitems, nlevels)
            )
            phi_star = phi[..., :-1].cumsum(axis=-1)
            gamma = pm.Deterministic(
                name='gamma',
                var=phi_star
            )
        param_list = [gamma]
        self.params.update({var.name: var for var in param_list})


class SlipPGRM(GRModel):
    def __init__(self, npersons, nitems, nlevels):
        super(SlipPGRM, self).__init__(npersons, nitems, nlevels)
        with self.model:
            phi = pm.Dirichlet(
                name='phi',
                a=np.ones(nlevels),
                shape=(1, nitems, nlevels)
            )
            phi_star = phi[..., :-1].cumsum(axis=-1)
            sigma = pm.Deterministic(
                name='sigma',
                var=phi_star
            )
        param_list = [sigma]
        self.params.update({var.name: var for var in param_list})


class FourPGRM(GRModel):
    def __init__(self, npersons, nitems, nlevels):
        super(FourPGRM, self).__init__(npersons, nitems, nlevels)
        with self.model:
            phi = pm.Dirichlet(
                name='phi',
                a=np.ones(2 * nlevels - 1),
                shape=(1, nitems, 2 * nlevels - 1)
            )
            # 1. drop the last term, which would make the top term 1 after
            #    cumsum.
            # 2. reshape into a 2 x L matrix for each item, each row
            #    corresponding to gamma and sigma, respectively
            # 3. cumulatively sum across the gamma->sigma, to ensure
            #    sigma > gamma
            # 4. cumulative gammas and sigmas across levels to ensure monotone
            phi_star = phi[..., :-1]
            phi_star = phi_star.reshape((1, nitems, 2, nlevels - 1))
            phi_star = phi_star.cumsum(axis=-2)
            phi_star = phi_star.cumsum(axis=-1)

            # first row is gamma, second is sigma
            gamma = pm.Deterministic(
                name='gamma',
                var=phi_star[..., 0, :]
            )
            sigma = pm.Deterministic(
                name='sigma',
                var=phi_star[..., 1, :]
            )
        param_list = [gamma, sigma]
        self.params.update({var.name: var for var in param_list})
