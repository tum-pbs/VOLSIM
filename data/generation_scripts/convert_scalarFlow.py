import numpy as np
import os
import scipy.ndimage

inDir = "ADD: path to downloaded and extracted scalarflow directory"
sims = 100

outDir = "data/128_test/scalarflow_step13/"
size = 128
timeStep = 13 #1,5,10,13
timeStart = 150 - 11*timeStep

for sim in range(sims):
    print("scalarflow/sim_%d" % sim)
    outPath = os.path.join(outDir, "sim_%06d" % sim)
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    results = []
    for i in range(11):
        loadPath = os.path.join(inDir, "sim_%06d/reconstruction/velocity_%06d.npz" % (sim, timeStart + i*timeStep))
        result = np.load(loadPath)['data']
        result = result[:,18:178,:,:] #cut off inflow

        zoom = [size/result.shape[0], size/result.shape[1], size/result.shape[2], 1]
        results += [ scipy.ndimage.zoom(result, zoom, order=1) ]

    outFile = os.path.join(outPath, "velocity_000000.npz")
    np.savez_compressed(outFile, np.stack(results, axis=0))

print("Done")
