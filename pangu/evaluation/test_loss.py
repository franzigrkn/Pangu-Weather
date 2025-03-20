import h5py
import numpy as np
import os
import time

from loss import WeatherBenchLoss_plain

# LOSS
loss_fn = WeatherBenchLoss_plain()

# print self.area
print(f"area: {loss_fn.area}")
print(f"Shape of area")