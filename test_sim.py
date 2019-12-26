import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
import models

shape = dict(npersons=1000, nitems=15, nlevels=5)

fixed = sim.GRMFixed(**shape)
fixed.default_params()
dataset = fixed.generate()
model_dict = dict(
    model2=models.TwoPGRM(**shape),
    model3l=models.GuessPGRM(**shape),
    model3u=models.SlipPGRM(**shape),
    model4=models.FourPGRM(**shape)
)
model_fits = {key: model.fit(dataset) for key, model in model_dict.items()}
model_bias = {key: sim.mean_abs_bias(trace, fixed.params) for key, trace in model_fits.items()}
