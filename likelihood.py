import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
from pymc3.distributions import Discrete, draw_values
from pymc3.math import sigmoid
from pymc3.theanof import floatX


def multidim_choice(probs, size=1):
    if np.any(probs < 0):
        raise ValueError("probs parameter must be non-negative.")

    # make sure that the size can be hstacked
    size = np.array(size)

    # create shape and generate uniform values
    shape = np.hstack([size, probs.shape[:-1], 1])
    unif = np.random.random(size=shape)

    # compare the uniform values to the cumulative probability
    # for each category, and return the integer representing
    # the lowest category that exceeds the uniform value
    cprobs = probs.cumsum(axis=-1)
    cprobs.shape = (1, ) * (unif.ndim - cprobs.ndim) + cprobs.shape
    sample = (unif < cprobs).argmax(axis=-1)

    return sample


class GRMLike(Discrete):
    def __init__(self, theta, alpha, kappa, gamma=0, sigma=1, *args, **kwargs):
        super(GRMLike, self).__init__(*args, **kwargs)

        self.param_list = []
        for var in [theta, alpha, kappa, gamma, sigma]:
            self.param_list.append(tt.as_tensor_variable(floatX(var)))

        self.params = {var.name: var for var in self.param_list}

        self.cprobst, self.probst = self.__init_probs()

        # Set number of categories
        self.k = tt.shape(self.probst)[-1]

        # Compute mode for each response category
        self.mode = tt.argmax(self.probst, axis=-1)

        # Numpy fancy indexing to allow observed data to index
        # probability tensor
        self.index = (tt.shape_padright(tt.arange(self.probst.shape[0])),
                      tt.shape_padleft(tt.arange(self.probst.shape[1])))

    def __init_probs(self):
        theta, alpha, kappa, gamma, sigma = self.param_list

        # Initialize probabilities
        # Compute the response cumulative probability functions
        resp_funcs = (sigma - gamma)*sigmoid(alpha*theta + kappa) + gamma

        # CDF for responses range from 0 to 1
        # reversing the category order to make the order of cprobs s.t.
        # index i is P(X >= i)
        cprobst = tt.concatenate([
            tt.ones_like(tt.shape_padright(resp_funcs[..., 0])),
            resp_funcs[..., ::-1],
            tt.zeros_like(tt.shape_padright(resp_funcs[..., 0]))
        ], axis=-1)

        # Discrete difference across response categories to get
        # marginal probabilities.
        # Identical to tt.extra_ops.diff
        probst = (cprobst[..., :-1] - cprobst[..., 1:])

        return cprobst, probst

    def logp(self, value):
        index = self.index + (value,)
        logliket = tt.log(self.probst[index])
        return logliket

    def random(self, point=None, size=1):
        probs, = draw_values(
            [self.probst], point=point, size=size
        )
        sample = multidim_choice(probs=probs, size=size)
        if size == 1:
            sample = sample[0]
        return sample

    def plot_cprobs(self, items=0):
        theta = self.param_list[0]
        if not type(theta) == np.ndarray:
            theta = theta.eval()
        for item in np.atleast_1d(items):
            plt.figure()
            lines = plt.plot(theta[:, 0, 0], self.cprobst[:, item].eval())
            plt.legend(handles=lines, labels=range(1, self.k.eval() + 2),
                       title=">= Response Level", loc="upper left")
        plt.show()

    def plot_probs(self, items=0):
        theta = self.param_list[0]
        if not type(theta) == np.ndarray:
            theta = theta.eval()
        for item in np.atleast_1d(items):
            plt.figure()
            lines = plt.plot(theta[:, 0, 0], self.probst[:, item].eval())
            plt.legend(handles=lines, labels=range(1, self.k.eval() + 2),
                       title=">= Response Level", loc="best")
        plt.show()
