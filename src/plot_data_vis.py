import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from volsim.params import *
from volsim.simulation_dataset import *

saveFolder = "results"


hp = Params(batch=1, gtDistMode="fit2", dataScaleInference=64, dataNormMin=0, dataNormMax=1, dataAugmentation=False, dataConvertMode="None")
print("Processing...")

dataSets = [
    SimulationDataset("Adv", ["data/64_example"], dataMode="full", filterTop=["advdiff"], printLevel="sim"),
    SimulationDataset("Iso", ["data/64_example"], dataMode="full", filterTop=["jhtdb_iso"], printLevel="sim"),
    SimulationDataset("Wav", ["data/64_example"], dataMode="full", filterTop=["waves"], printLevel="sim"),
]

if not os.path.isdir(saveFolder):
    os.mkdir(saveFolder)

dataLoaders = []
trans = TransformsInference(normalize="single", order=0, params=hp)
for dataSet in dataSets:
    dataSet.setDataTransform(trans)
    #dataSet.printDatasetInfo()
    dataSet.loadDistanceCoefficients(hp.gtDistMode)
    dataLoaders += [DataLoader(dataSet, batch_size=hp.batch, shuffle=False, num_workers=1)]

for dataLoader, dataSet in zip(dataLoaders, dataSets):
    print(dataSet.name)
    fig, ax = plt.subplots(2,11, figsize=(7,1.2), dpi=200, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.03, hspace=0.06)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax[0,0].set_ylabel("Slice")
    ax[1,0].set_ylabel("Mean")
    for i in range(11):
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])

    for s, sample in enumerate(dataLoader, 0):

        plt.title("%s\ndistance to ref according to similarity model" % sample["path"][0])

        data = sample["data"].numpy()[0]
        dist = sample["distance"][0]

        dataMean = np.mean(data, axis=3)
        dMin = np.min(dataMean)
        dMax = np.max(dataMean)
        dataMean = (dataMean - dMin) / (dMax - dMin)

        combinedSlice = []
        combinedMean = []
        for i in range(data.shape[0]):
            s = data[i]
            rot = np.rot90(s[:, :, int(s.shape[2]/2)])
            ax[0,i].imshow(rot)

            rot = np.rot90(dataMean[i])
            ax[1,i].imshow(rot)
            if i == 0:
                ax[1,i].set_xlabel("ref")
            else:
                ax[1,i].set_xlabel("%0.2f" % dist[i-1])

    plt.savefig('%s/DataVis_%s.png' % (saveFolder, dataSet.name), bbox_inches='tight', pad_inches = 0)
print("Images saved to: %s" % (saveFolder))