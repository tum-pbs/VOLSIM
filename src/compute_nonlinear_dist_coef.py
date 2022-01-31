import numpy as np
import torch
from scipy.optimize import curve_fit
import hashlib, json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from volsim.simulation_dataset import *
from volsim.metrics import *

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42

#torch.manual_seed(2)


inputScale = 64

savePDF = "NonlinearCoefficients"
writeJSON = True
plot = False

corr = Metric("CORR")
mse = Metric("MSE")

dataSets = [
    SimulationDataset("Train", ["data/%d_train" % inputScale], dataMode="full", printLevel="sim"),#, overfitCount=2),
    SimulationDataset("Test", ["data/%d_test" % inputScale], dataMode="full", printLevel="sim"),#, overfitCount=2),
]
plotIdxs = [0,1]


def logFuncA(x, coef):
    return np.log((10**coef) * x + 1) / np.log(10**coef + 1)

def logFuncB(x, coef1, coef2):
    return np.log((10**coef1) * x + 1) / coef2

def logFuncC(x, coef):
    return np.log(coef * x + 1) / np.log(coef + 1)

transforms = TransformsInference(normalize="single", order=0, params=Params(dataScaleInference=inputScale, dataConvertMode="none", dataNormQuant=1.0, dataNormMin=-1.0, dataNormMax=1.0))
dataLoaders = []

for dataSet in dataSets:
    dataSet.setDataTransform(transforms)
    dataSet.printDatasetInfo()
    dataLoaders += [DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=4)]



# Caching
hashStr = hashlib.sha1(savePDF.encode()).hexdigest()[:6]

cachePath = ".cache"
if not os.path.isdir(cachePath):
    os.makedirs(cachePath)
files = os.listdir(cachePath)
cached = None

for f in files:
    if hashStr in f:
        cached = os.path.join(cachePath, f)
        break

if cached:
    print("Re-using cached coefficients: %s\n" % (hashStr))
    loadDict = torch.load(cached)
    allDistsCorr1 = loadDict["allDistsCorr1"]
    allDistsCorr2 = loadDict["allDistsCorr2"]
    allDistsMSE = loadDict["allDistsMSE"]
    allCoefs1 = loadDict["allCoefs1"]
    allCoefs2 = loadDict["allCoefs2"]
else:
    print("Starting coefficient computation...")

    allDistsCorr1 = {}
    allDistsCorr2 = {}
    allDistsMSE = {}
    allCoefs1 = {}
    allCoefs2 = {}

    for d in range(len(dataSets)):
        coefs1 = {}
        coefs2 = {}

        for s, sample in enumerate(dataLoaders[d], 0):
            key = sample["path"][0]

            sample["idxMin"] = 0
            sample["idxMax"] = 10
            distCorr = corr(sample)[0].numpy()
            distMSE = mse(sample)[0].numpy()
            allDistsMSE[key] = distMSE

            xVals = sample["distance"][0, 0:10].numpy()

            # fit 1
            dMax = np.max(distCorr)
            distCorr1 = distCorr / dMax
            allDistsCorr1[key] = distCorr1

            optCoef1, _ = curve_fit(logFuncA, xVals, distCorr1, p0=0.1)
            coefs1[key] = [optCoef1[0]]
            allCoefs1[key] = [optCoef1[0]]

            # fit 2
            dMax = np.max(distCorr)
            dMin = np.min(distCorr)
            distCorr2 = (distCorr - dMin) / (dMax - dMin)
            allDistsCorr2[key] = distCorr2

            optCoef2, _ = curve_fit(logFuncA, xVals, distCorr2, p0=0.1)
            coefs2[key] = [optCoef2[0]]
            allCoefs2[key] = [optCoef2[0]]

            print(sample["path"])
            #print(dist)
            #print(xVals)
            #print(logFunc(xVals, optCoef))
            #print(optCoef, var)

        if writeJSON:
            #jsonFile = "%s/distance_coefficients_old.json" % (dataSets[d].dataDirs[0])
            #json.dump(coefs1, open(jsonFile, "w"), indent=2)

            jsonFile = "%s/distance_coefficients.json" % (dataSets[d].dataDirs[0])
            json.dump(coefs2, open(jsonFile, "w"), indent=2)

            print("Coefficient dictionaries written to %s" % dataSets[d].dataDirs[0])
        print("")

    saveDict = {}
    saveDict["allDistsCorr1"] = allDistsCorr1
    saveDict["allDistsCorr2"] = allDistsCorr2
    saveDict["allDistsMSE"] = allDistsMSE
    saveDict["allCoefs1"] = allCoefs1
    saveDict["allCoefs2"] = allCoefs2
    torch.save(saveDict, os.path.join(cachePath, "nonlinearCoefficients_%s.cache" % hashStr))


if plot:
    # Plotting
    pdf = PdfPages("results/" + savePDF + ".pdf")

    for d in range(len(dataSets)):
        for idx in [0,1]:
            sample = dataSets[d][idx]
            key = sample["path"]
            data = sample["data"].permute(0,2,3,4,1).cpu().numpy()

            xVals = np.arange(0.1,1.01,0.1)
            distCorr1 = allDistsCorr1[key]
            distCorr2 = allDistsCorr2[key]
            distMSE = allDistsMSE[key]
            optCoef1 = allCoefs1[key]
            optCoef2 = allCoefs2[key]

            # plot distance curves and coefficients
            fig, ax = plt.subplots(1,1, figsize=(7.0,3.0), dpi=200, tight_layout=True)

            ax.plot(xVals, xVals, label="Linear distances", color="0.6", marker=".")
            ax.plot(xVals, distMSE / np.max(distMSE), label="MSE / max(MSE)", color="0.1", marker=".")

            samples = np.arange(0,1,0.01)
            distCorrFine1 = logFuncA(samples, optCoef1[0])
            distCorrCoarse1 = logFuncA(xVals, optCoef1[0])
            ax.plot(xVals, distCorr1, label="Correlation / max(Correlation)", color="r", marker=".")
            ax.plot(samples, distCorrFine1, label="Fit 1: $log(10^c * x + 1) / log(10^c + 1)$", color="orange", linestyle="dotted")
            ax.scatter(xVals, distCorrCoarse1, label="", color="orange", marker="x")

            distCorrFine2 = logFuncA(samples, optCoef2[0])
            distCorrCoarse2 = logFuncA(xVals, optCoef2[0])
            ax.plot(xVals, distCorr2, label="Correlation (normalized)", color="darkgreen", marker=".")
            ax.plot(samples, distCorrFine2, label="Fit 2: $log(10^c * x + 1) / log(10^c + 1)$", color="lightgreen", linestyle="dotted")
            ax.scatter(xVals, distCorrCoarse2, label="", color="lightgreen", marker="x")

            folders = key.split("/")
            ax.set_title("Fitted functions for %s" % (folders[1]))
            ax.set_xlabel("Simulation steps (normalized)")
            ax.set_ylabel("Distance")
            ax.grid(True)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left", bbox_to_anchor=(1,1))

            pdf.savefig(fig, bbox_inches='tight')


            # plot data examples
            fig, axs = plt.subplots(1, 11, figsize=(13, 1.3), dpi=200, sharey=True)
            fig.subplots_adjust(wspace=0.02, hspace=0.10)
            axs[0].set_ylabel("Data (z-mean)")
            fig.suptitle("%s - %s" % (folders[1], folders[2]))

            for i in range(data.shape[0]):
                ax = axs[i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("$t=%d$" % i)

                sliced = data[i]

                mean = np.mean(sliced, axis=3)
                dMin = np.min(mean)
                dMax = np.max(mean)
                mean = (mean - dMin) / (dMax - dMin)
                ax.imshow(mean, vmin=0, vmax=1, interpolation="nearest")

            pdf.savefig(fig, bbox_inches='tight')

    pdf.close()
    print("Plot written to %s" % ("results/" + savePDF + ".pdf"))




