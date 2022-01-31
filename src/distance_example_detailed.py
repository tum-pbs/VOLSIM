## IMPORTS
import os
import numpy as np
import scipy.stats.stats as sciStats
from volsim.metrics import *
from volsim.distance_model import *
from volsim.simulation_dataset import *
np.set_printoptions(precision=3, floatmode="fixed", linewidth=120, suppress=True)

## INITIALIZATION
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
useGPU = True
printDistances = False
gtDistanceMode = "fit2"         # "fit2" for ground truth distances from similarity model, "lin" for linearly increasing distances
correlationMode = "spearman"    # "spearman" or "pearson" for different correlation computation methods
evaluationSliceSize = 55        # can be reduced to decrease GPU memory usage, should be in [1,5,11,55]

names = []
models = []
transforms = []

names += ["VolSiM"]
models += [DistanceModel.load("models/VolSiM.pth", useGPU=useGPU)]
transforms += [TransformsInference(normalize="single", order=0, params=models[0].hp)]

names += ["MSE"]
models += [Metric("MSE")]
transforms += [TransformsInference("single", 0, Params(dataScaleInference=-1, dataNormQuant=1.0, dataNormMin=-1.0, dataNormMax=1.0))]

dataSets = [
    SimulationDataset("Adv", ["data/64_example"], dataMode="full", filterTop=["advdiff"], printLevel="sim"),
    SimulationDataset("Iso", ["data/64_example"], dataMode="full", filterTop=["jhtdb_iso"], printLevel="sim"),
    SimulationDataset("Wav", ["data/64_example"], dataMode="full", filterTop=["waves"], printLevel="sim"),
]


## DISTANCE COMPUTATION
for dataset in dataSets:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    dataset.loadDistanceCoefficients(gtDistanceMode)
    print("")
    print("--------------------------------")
    print("Processing %s data sequence:" % (dataset.name))

    with torch.no_grad():
        for m in range(len(models)):
            dataset.setDataTransform(transforms[m])

            for s, sample in enumerate(dataloader, 0):
                slicingCount = sample["indexA"].shape[1]

                distGt = []
                distPred = []
                for i in range(0, slicingCount, evaluationSliceSize):
                    sample["idxMin"] = i
                    sample["idxMax"] = min(i + evaluationSliceSize, slicingCount)

                    target = sample["distance"][:,sample["idxMin"]:sample["idxMax"]]
                    prediction = models[m](sample)

                    distGt += [target.cpu().numpy()]
                    distPred += [prediction.cpu().numpy()]

                gt = np.concatenate(distGt, axis=1)[0]
                pred = np.concatenate(distPred, axis=1)[0]
                stacked = np.stack([gt, pred], axis=0)

                correlation = None
                if correlationMode == "pearson":
                    correlation = np.corrcoef(stacked)[0,1]
                elif correlationMode == "spearman":
                    correlation = sciStats.spearmanr(gt, pred)[0]

                if printDistances:
                    print("\t" + "%s distances (ground truth, prediction):" % (names[m]))
                    print("\t\t" + str(gt).replace('\n','\n\t\t'))
                    print("\t\t" + str(pred).replace('\n','\n\t\t'))

                print("\t" + "%s distance correlation (%s) with ground truth:  %0.3f" % (names[m], correlationMode, correlation))

                if m != len(models)-1:
                    print()

