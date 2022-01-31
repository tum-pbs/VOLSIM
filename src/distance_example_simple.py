## IMPORTS
import numpy as np
from volsim.metrics import *
from volsim.distance_model import *


## INITIALIZATION
useGPU = False

modelVolsim = DistanceModel.load("models/VolSiM.pth", useGPU=useGPU)
modelMSE = Metric("MSE")

data = np.load("data/64_example/waves/sim_000000/flags_000000.npz")['arr_0']
print("Loading data sequence of shape " + str(data.shape))

# batched comparison of sequences elements:
# element 0 is compared to 9 and element 3 compared to 5
arr1 = np.stack([data[0], data[3]])
arr2 = np.stack([data[9], data[5]])

## DISTANCE COMPUTATION
distVolsim = modelVolsim.computeDistance(arr1, arr2, normalize=True, interpolate=True)
distMSE = modelMSE.computeDistance(arr1, arr2, normalize=False, interpolate=True)

print()
print("Distances:")
print("VolSiM: %0.4f, %0.4f" % (distVolsim[0], distVolsim[1]))
print("MSE: %0.4f, %0.4f" % (distMSE[0], distMSE[1]))

# distance output should look like this on GPU and CPU:
# VolSiM: 0.7140, 0.4329
# MSE: 0.0061, 0.0065