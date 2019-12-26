import numpy as np
from scipy.stats import norm

from likelihood import GRMLike


class _Unpickling(object):
    pass


class LikeParams(dict):
    def __new__(cls, name, *args, **kwargs):
        if name is _Unpickling:
            instance = super(LikeParams, cls).__new__(cls, *args, **kwargs)
            return instance
        else:
            instance = super(LikeParams, cls).__new__(cls, *args, **kwargs)
            return instance


    def __getnewargs__(self):
        return _Unpickling,

    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels - 1)
        keys = ('theta', 'alpha', 'kappa', 'gamma', 'sigma')
        def_dict = {key: np.zeros(self.shape) for key in keys}
        def_dict['alpha'] += 1
        def_dict['sigma'] += 1
        super(LikeParams, self).__init__(**def_dict)

    def __getstate__(self):
        self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

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
        value = np.atleast_1d(value)
        self.alpha = self.__broadcast_alpha(value)

    def set_kappa(self, value):
        value = np.atleast_1d(value)
        if not np.all(np.diff(value, axis=-1) > 0):
            raise ValueError("Kappa must be strictly increasing.")
        self.kappa = self.__broadcast_level(value)

    def set_gamma(self, value):
        value = np.atleast_1d(value)
        if not np.all(np.diff(value, axis=-1) >= 0):
            raise ValueError("Gamma must be monotonically increasing.")
        self.gamma = self.__broadcast_level(value)

    def set_sigma(self, value):
        value = np.atleast_1d(value)
        if not np.all(np.diff(value, axis=-1) > 0):
            raise ValueError("Sigma must be monotonically increasing.")
        self.sigma = self.__broadcast_level(value)

    def set_theta(self, value):
        value = np.atleast_1d(value)
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

    def norm_theta(self):
        self.set_theta(norm().rvs(size=self.shape[0]))

    def line_theta(self):
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
    def __new__(cls, name, *args, **kwargs):
        if name is _Unpickling:
            return object.__new__(cls)
        else:
            instance = super(GRMFixed, cls).__new__(cls)
            return instance

    def __getnewargs__(self):
        return _Unpickling,

    def __init__(self, npersons, nitems, nlevels):
        self.shape = (npersons, nitems, nlevels - 1)
        self.params = LikeParams(npersons, nitems, nlevels)
        self.__likelihood = GRMLike.dist(**self.params)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def default_params(self):
        norm_quants = norm.ppf(np.linspace(1, 0, self.shape[2] + 2)[1:-1])
        self.params.set_diffs(norm_quants, 3)
        self.params.set_gamma(np.linspace(0.1, 0.3, self.shape[2]))
        self.params.set_sigma(np.linspace(0.7, 0.9, self.shape[2]))
        self.params.norm_theta()

    def check_recompile(self):
        old_dict = self.__likelihood.params
        old_pars = {name: var.eval() for name, var in old_dict.items()}
        dict_zip = ((old_pars[key], self.params[key]) for key in self.params)
        return all(np.all(x == y) for x, y in dict_zip)

    @property
    def likelihood(self):
        if not self.check_recompile():
            self.__likelihood = GRMLike.dist(**self.params)
        return self.__likelihood

    def generate(self, size=1, seed=None):
        np.random.seed(seed)
        print("Seed Number: ", seed)
        self.dataset = self.likelihood.random(size=size)

        return self.dataset


def stat_params(trace, varnames, stat):
    stat_dict = {var: stat(trace[var], axis=0) for var in varnames}
    if 'alpha' in varnames and 'gamma' in varnames and 'sigma' in varnames:
        scale = stat_dict['sigma'] - stat_dict['gamma']
        stat_dict['slope'] = stat_dict['alpha']*scale
    return stat_dict


def abs_bias(trace, params, varnames=None, stat=np.mean):
    if varnames is None:
        varnames = params.keys()
    elif not all(x in params.keys() for x in varnames):
        diff_keys = set(varnames) - set(params.keys())
        raise ValueError("Params not found in GRM params: %s"
                         % diff_keys)

    stat_dict = stat_params(trace, varnames, stat)
    bias_dict = {}
    for var in varnames:
        if var == 'theta':
            true_val = params[var][:, :1, :1]
        elif var == 'alpha':
            true_val = params[var][:1, :, :1]
        else:
            true_val = params[var][:1, :, :]
        bias_dict[var] = abs(stat_dict[var] - true_val)

    if 'slope' in stat_dict:
        scale = params['sigma'] - params['gamma']
        bias_dict['slope'] = params['alpha']*scale
        bias_dict['slope'] = abs(stat_dict['slope'] - bias_dict['slope'][:1])

    return bias_dict


def mean_abs_bias(trace, params, varnames=None, stat=np.mean):
    bias_dict = abs_bias(trace, params, varnames, stat)
    for var in bias_dict:
        bias_dict[var] = bias_dict[var].mean()
    return bias_dict
