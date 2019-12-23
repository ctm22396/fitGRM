import numpy as np
import pymc3 as pm
from pymc3.distributions.transforms import ordered

from .likelihood import GRMLike


class GRModel(object):
    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels - 1)
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

    def fit(self, y):
        self.y = y
        with self.model:
            pi_hat = GRMLike(
                'pi_hat',
                **self.params,
                observed=y
            )
        self.likelihood = pi_hat


class TwoPGRM(GRModel):
    pass


class ThreePGRM(GRModel):
    def __init__(self, npersons, nitems, nlevels):
        super(ThreePGRM, self).__init__(npersons, nitems, nlevels)
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


def lower_3PGRM(dataset):
    npersons, nitems = dataset.shape
    nlevels = np.unique(dataset).size

    with pm.Model() as model:
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

        params = {var.name: var for var in [theta, alpha, kappa, gamma]}
        pi_hat = GRMLike(
            'pi_hat',
            **params,
            observed=dataset
        )
    return model, params, pi_hat


def upper_3PGRM(dataset):
    npersons, nitems = dataset.shape
    nlevels = np.unique(dataset).size

    with pm.Model() as model:
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

        params = {var.name: var for var in [theta, alpha, kappa, sigma]}
        pi_hat = GRMLike(
            'pi_hat',
            **params,
            observed=dataset
        )
    return model, params, pi_hat


def basic_PGRM(dataset):
    npersons, nitems = dataset.shape
    nlevels = np.unique(dataset).size

    with pm.Model() as model:
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

        params = {var.name: var for var in [theta, alpha, kappa]}
        pi_hat = GRMLike(
            'pi_hat',
            **params,
            observed=dataset
        )
    return model, params, pi_hat
