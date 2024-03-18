"""Script to test WaveNet NN."""

import numpy as np
import run_wavenet as rwn


IMAX = 128
NUM_COL = 4

# Generate the four input tensors and populate with random data
wind = np.random.randn(IMAX * NUM_COL, 40)
lat = np.random.randn(IMAX * NUM_COL, 1)
ps = np.random.randn(IMAX * NUM_COL, 1)
Y_out = np.zeros((IMAX * NUM_COL, 40))

# Initialise and run the model
model = rwn.initialize()
Y_out = rwn.compute_reshape_drag(model, wind, lat, ps, Y_out, NUM_COL)
