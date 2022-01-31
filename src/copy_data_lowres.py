# conversion of simulation data to lower resolutions

import numpy as np
import os, shutil
import scipy.ndimage

inDir = "data/128_train/"
outDir = "data/64_train/"
res = 64

for root, dirs, files in os.walk(inDir):
    for fileName in files:
        filePath = os.path.join(root, fileName)
        outPath = filePath.replace(inDir, outDir)
        outFolder = os.path.dirname(outPath)

        if not os.path.exists(outFolder):
            os.makedirs(outFolder)

        if os.path.splitext(fileName)[1] == ".npz":
            data = np.load(filePath)['arr_0']

            zoom = [1, res/data.shape[1], res/data.shape[2], res/data.shape[3], 1]
            dataLow = scipy.ndimage.zoom(data, zoom, order=1)

            np.savez_compressed(outPath, dataLow )
            print("%s  %s  %s" % (filePath,str(data.shape),str(dataLow.shape)))

        else:
            shutil.copy(filePath, outPath)

