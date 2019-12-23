from .likelihood import GRMLike

import numpy as np


class LikeParams(dict):
    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels - 1)
        keys = ('theta', 'alpha', 'kappa', 'gamma', 'sigma')
        def_dict = {key: np.zeros(self.shape) for key in keys}
        def_dict['alpha'] += 1
        def_dict['sigma'] += 1
        super(LikeParams, self).__init__(**def_dict)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        if attr in self.keys():
            if type(value) == np.ndarray:
                if not value.shape == self.shape:
                    raise ValueError("Must set with array of same shape.")
                else:
                    self[attr] = value
            else:
                raise ValueError("Must set value with array of same shape.")
        else:
            super(LikeParams, self).__setattr__(attr, value)

    def __broadcast_alpha(self, value):
        dim = value.ndim
        if dim > 1:
            raise("Alpha must be constant or one-dimensional,"
                  "fixed across levels>")
        new_value = np.atleast_3d(value)
        reps = np.array(self.shape) / np.array(new_value.shape)
        if not all(reps == reps.astype(int)):
            raise ValueError("Shape of supplied values must "
                             "divide the original shape")
        out_value = np.tile(new_value, reps=reps.astype(int))
        return(out_value)

    def __broadcast_level(self, value):
        shape = value.shape
        dim = value.ndim
        new_shape = (1,)*(3 - dim) + shape
        new_value = value.reshape(new_shape)

        reps = np.array(self.shape) / np.array(new_shape)
        if not all(reps == reps.astype(int)):
            raise ValueError("Shape of supplied values must "
                             "divide the original shape")
        out_value = np.tile(new_value, reps=reps.astype(int))
        return out_value

    def set_alpha(self, value):
        value = np.array(value)
        self.alpha = self.__broadcast_alpha(value)

    def set_kappa(self, value):
        value = np.array(value)
        if not np.all(np.diff(value, axis=-1) > 0):
            raise ValueError("Kappa must be strictly increasing.")
        self.kappa = self.__broadcast_level(value)

    def set_gamma(self, value):
        value = np.array(value)
        if not np.all(np.diff(value, axis=-1) >= 0):
            raise ValueError("Gamma must be monotonically increasing.")
        self.gamma = self.__broadcast_level(value)

    def set_sigma(self, value):
        value = np.array(value)
        if not np.all(np.diff(value, axis=-1) > 0):
            raise ValueError("Sigma must be monotonically increasing.")
        self.sigma = self.__broadcast_level(value)

    def set_theta(self, value):
        value = np.array(value)
        shape = value.shape
        if any(np.array(shape[1:]) > 1):
            raise ValueError("Theta value must reduce to one dimension")
        dim = value.ndim
        new_shape = shape + (1,)*(3 - dim)
        self.theta += value.reshape(new_shape)

    def set_diffs(self, diffs, alphas=1):
        diffs = np.atleast_2d(diffs)
        alphas = np.atleast_1d(alphas)
        kappas = -diffs * alphas[:, None]
        self.set_alpha(alphas)
        self.set_kappa(kappas)

    def auto_theta(self):
        if not np.all(self.kappa == 0):
            diffs = -self.kappa/self.alpha
            lower = diffs.min()
            upper = diffs.max()
            mid = (upper + lower)/2
            span = upper - lower
            spread = 1.5 * span/2 * np.array([-1, 1]) + mid
            self.set_theta(np.linspace(spread[0], spread[1], self.shape[0]))
        else:
            raise ValueError("Kappa not set yet.")

    def set_asymps(self, value):
        value = np.array(value)
        self.set_gamma(value[..., ::2])
        self.set_sigma(value[..., 1::2])


class GRMFixed(object):
    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels - 1)
        self.params = LikeParams(npersons, nitems, nlevels)
        self.__likelihood = GRMLike.dist(**self.params)

    def default_params(self):
        self.params.set_diffs(np.linspace(2, -2, self.shape[2]), 3)
        self.params.set_gamma(np.linspace(0.1, 0.3, self.shape[2]))
        self.params.set_sigma(np.linspace(0.7, 0.9, self.shape[2]))
        self.params.auto_theta()

    def check(self):
        old_pars = {name: var.eval() for name, var in self.__likelihood.params}
        return old_pars == self.params

    @property
    def likelihood(self):
        if not self.checK():
            self.__likelihood = GRMLike.dist(**self.params)
        return self.__likelihood

    def generate(self, size, seed=None):
        np.random.seed(seed)
        print("Seed Number: ", seed)
        self.dataset = self.likelihood.random(size=size)

        return self.dataset
