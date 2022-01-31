import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import os, json
import logging
import scipy.ndimage

from volsim.params import *


class SimulationDataset(Dataset):

    def __init__(self, name:str, dataDirs:list, dataMode:str="scalar", filterTop:list=[], excludeFilterTop:bool=False,
                filterSimDict:dict={}, excludeFilterSim:bool=False, filterFrameDict:dict={},
                excludeFilterFrame:bool=False, split:dict={}, overfitCount:int=-1,
                printLevel:str="none", logLevel:str="full", noTransformWarning:bool=False):
        """
        Data loader for simulation data 

        :param name: name of the dataset
        :param dataDirs: list of paths to data directories
        :param dataMode: treatment of multi channel data like velocities: split to scalar fields, keep channels, or both ["scalar", "full", "both"]
        :param filterTop: filter simulation names.
        :param excludeFilterTop: mode for filterTop (exclude or include).
        :param filterSimDict: filter simulation name (key) by list of simulation seeds (value), e.g. "smoke" : ["000001"].
        :param excludeFilterSim: mode for filterSimDict (exclude or include).
        :param filterFrameDict: filter simulation name (key) by list of corresponding simulation frames (value), e.g. "smoke" : ["000001"].
        :param excludeFilterFrame: mode for filterFrameDict (exclude or include)
        :param split: evaluate data set in splits (for validation or test set) via tensorbard tag (key) and search string (value)
        :param overfitCount: subsamples the dataset to create a smaller version with overfitCount many random samples
        :param printLevel: print contents of the dataset on initialization ["none", "sim", "seed", "full"]
        :param logLevel: log contents of the dataset on initialization ["none", "sim", "seed", "full"]
        """
        assert (dataMode in ["scalar", "full", "both"]), "Invalid data mode!"
        assert (printLevel in ["none", "sim", "seed", "full"]), "Invalid print level!"
        self.transform = None
        self.name = name
        self.dataDirs = dataDirs
        self.dataMode = dataMode
        self.filterTop = filterTop
        self.excludeFilterTop = excludeFilterTop
        self.filterSimDict = filterSimDict
        self.excludeFilterSim = excludeFilterSim
        self.filterFrameDict = filterFrameDict
        self.excludeFilterFrame = excludeFilterFrame
        self.split = split
        self.overfitCount = overfitCount
        self.noTransformWarning = noTransformWarning

        self.summaryPrint = []
        self.summaryLog = []
        self.summaryPrint += ["Dataset " + name + " at " + str(dataDirs)]
        self.summaryLog   += ["Dataset " + name + " at " + str(dataDirs)]
        self.summaryPrint += [self.getFilterInfoString()]
        self.summaryLog   += [self.getFilterInfoString()]

        # BUILD SIMULATION FILE LIST
        self.dataPaths = []
        self.dataPathModes = []

        for dataDir in dataDirs:
            topDirs = os.listdir(dataDir)
            topDirs.sort()

            # top level folders
            for topDir in topDirs:
                if filterTop:
                    # continue for simulations when excluding or including according to filter
                    if excludeFilterTop == any( item in topDir for item in filterTop ):
                        continue

                simDir = os.path.join(dataDir, topDir)
                if not os.path.isdir(simDir):
                    continue
                sims = os.listdir(simDir)
                sims.sort()

                if printLevel == "sim":
                    self.summaryPrint += ["Loaded: " + simDir.replace(dataDir + "/", "")]
                if logLevel == "sim":
                    self.summaryLog   += ["Loaded: " + simDir.replace(dataDir + "/", "")]

                # sim_000000 folders
                for sim in sims:
                    if filterSimDict:
                        filterSims = []
                        for key, value in filterSimDict.items():
                            if key in topDir:
                                filterSims = value
                                break
                        assert (filterSims != []), "Keys in filterSimDict don't match dataDir structure!"

                        # continue for sims when excluding or including according to filter
                        if excludeFilterSim == any( item in sim for item in filterSims ):
                            continue

                    currentDir = os.path.join(simDir, sim)
                    frames = os.listdir(currentDir)
                    frames.sort()

                    if printLevel == "seed":
                        self.summaryPrint += ["Loaded: " + currentDir.replace(dataDir + "/", "")]
                    if logLevel == "seed":
                        self.summaryLog   += ["Loaded: " + currentDir.replace(dataDir + "/", "")]

                    # individual simulation frames xyz.npz and logging files/dirs
                    for frame in frames:
                        currentFrame = os.path.join(currentDir, frame)
                        if os.path.isdir(currentFrame):
                            continue
                        elif os.path.splitext(frame)[1] != ".npz":
                            continue

                        if filterFrameDict:
                            filterFrames = []
                            for key, value in filterFrameDict.items():
                                if key in topDir:
                                    filterFrames = value
                                    break
                            assert (filterFrames != []), "Keys in filterFrameDict don't match dataDir structure!"

                            # continue for frames when excluding or including according to filter
                            if excludeFilterFrame == any( item in frame for item in filterFrames ):
                                continue

                        if printLevel == "full":
                            self.summaryPrint += ["Loaded: " + currentFrame.replace(dataDir + "/", "")]
                        if logLevel == "full":
                            self.summaryLog   += ["Loaded: " + currentFrame.replace(dataDir + "/", "")]

                        if not "velocity" in currentFrame:
                            self.dataPaths.append(currentFrame)
                            self.dataPathModes.append(-1)
                        else:
                            if self.dataMode in ["full", "both"]:
                                self.dataPaths.append(currentFrame)
                                self.dataPathModes.append(-1)
                            if self.dataMode in ["scalar", "both"]:
                                self.dataPaths.append(currentFrame)
                                self.dataPathModes.append(0)
                                self.dataPaths.append(currentFrame)
                                self.dataPathModes.append(1)
                                self.dataPaths.append(currentFrame)
                                self.dataPathModes.append(2)


        # BUILD INDEX LOOK-UP TABLES
        data = np.load(self.dataPaths[0])['arr_0'] # load first item to check data dimensions
        temp = np.linspace(0.0, 1.0, num=data.shape[0])

        indexA = []
        indexB = []
        distance = []
        for i in range(data.shape[0]):
            for j in range(i+1,data.shape[0]):
                indexA.append(i)
                indexB.append(j)
                distance.append(temp[j] - temp[i])
        self.indexA = np.array(indexA, dtype=np.int32)
        self.indexB = np.array(indexB, dtype=np.int32)
        self.distance = np.array(distance, dtype=np.float32)

        self.gtDistMode = "lin"
        self.distanceCoefficients = {}

        self.summaryPrint += ["Length: %d\n" % len(self.dataPaths)]
        self.summaryLog   += ["Length: %d\n" % len(self.dataPaths)]

        if self.overfitCount > 0:
            randGen = np.random.RandomState(torch.random.initial_seed() % 4294967295)
            temp = np.arange(len(self.dataPaths))
            randGen.shuffle(temp)
            self.overfitSamples = temp[:overfitCount]
            self.summaryPrint += ["Overfit dataset of %d samples." % self.overfitCount]
            self.summaryPrint += ["Permutation: %s\n" % str(self.overfitSamples)]
            self.summaryLog   += ["Overfit dataset of %d samples.\n" % self.overfitCount]
            self.summaryLog   += ["Permutation: %s\n" % str(self.overfitSamples)]


    def __len__(self) -> int:
        if self.overfitCount > 0:
            return self.overfitSamples.shape[0]
        else:
            return len(self.dataPaths)


    def __getitem__(self, idx:int) -> dict:
        if self.overfitCount > 0:
            filePath = self.dataPaths[self.overfitSamples[idx]]
            channelSlice = self.dataPathModes[self.overfitSamples[idx]]
        else:
            filePath = self.dataPaths[idx]
            channelSlice = self.dataPathModes[idx]

        data = np.load(filePath)['arr_0'] #shape 11x128x128x128x1 or x3 for vel
        data = data.astype(np.float32)
        if channelSlice >= 0:
            data = data[..., channelSlice] # single channel for dataModes scalar and both
            data = data[..., np.newaxis]

        newPath = filePath
        newPath = newPath.replace("data/", "")

        if self.distanceCoefficients and self.gtDistMode != "lin":
            coef = self.distanceCoefficients[newPath]

            if self.gtDistMode in ["fit1", "fit2"]:
                dist = np.log((10**coef[0]) * self.distance + 1) / np.log(10**coef[0] + 1)
            else:
                raise ValueError("Invalid ground truth distance mode")
        else:
            dist = self.distance
        dist = dist.astype(np.float32)

        sample = {"path" : newPath, "data": data, "indexA" : self.indexA,
                "indexB" : self.indexB, "distance" : dist}

        if self.transform:
            sample = self.transform(sample)
        else:
            if not self.noTransformWarning:
                print("WARNING: no data transforms are employed!")

        return sample



    def printDatasetInfo(self):
        print('\n'.join(self.summaryPrint))
        logging.info('\n'.join(self.summaryLog))

    def getFilterInfoString(self) -> str:
        s  = "%s Data Filter Setup: \n" % (self.name)
        s += "\tdataDirs: %s\n" % (str(self.dataDirs))
        s += "\tfilterTop: %s  exclude: %s\n" % (str(self.filterTop), self.excludeFilterTop)
        s += "\ffilterSim: %s  exclude: %s\n" % (str(self.filterSimDict), self.excludeFilterSim)
        s += "\tfilterFrame: %s  exclude: %s" % (str(self.filterFrameDict), self.excludeFilterFrame)
        return s

    def setDataTransform(self, transform:callable):
        self.transform = transform

    def loadDistanceCoefficients(self, gtDistMode:str):
        assert (gtDistMode in ["lin", "fit2"]), "Invalid ground truth distance mode!"
        self.gtDistMode = gtDistMode

        if gtDistMode != "lin":
            for dataDir in self.dataDirs:
                #mode = gtDistMode[-1]

                jsonFile = "%s/distance_coefficients.json" % (dataDir)
                coeff = json.load(open(jsonFile))
                self.distanceCoefficients.update(coeff)




# ------------------------------------------------- 
# TRANSFORMS TO APPLY TO THE DATA
# -------------------------------------------------

# combines normalization, flip, 90 degree rotation, channel swap, crop, to tensor 
class TransformsTrain(object):
    def __init__(self, permuteIndex:bool, normalize:str, params:Params):
        self.permuteIndex = permuteIndex
        self.normalize = normalize
        self.flip = params.dataAugmentation
        self.rot90 = params.dataAugmentation
        self.channelSwap = params.dataAugmentation
        self.crop = params.dataCrop
        self.cropRandom = params.dataCropRandom
        self.ignoreCrop = False
        self.normQuantile = params.dataNormQuant
        self.normMin = params.dataNormMin
        self.normMax = params.dataNormMax
        self.rotAxes = [(1,2), (1,3), (2,3)]
        self.cutoff = params.dataCutoffIndex

        # seeding once for single thread data loading
        self.randGen = np.random.RandomState(torch.random.initial_seed() % 4294967295)

        assert self.normalize in ["none", "length", "channel", "single"]

    def __call__(self, sample:dict):
        # seeding in every call for multi thread data loading
        if torch.utils.data.get_worker_info():
            self.randGen = np.random.RandomState(torch.utils.data.get_worker_info().seed % 4294967295)

        data = sample["data"]
        indexA = sample["indexA"]
        indexB = sample["indexB"]
        distance = sample["distance"]

        # quantile normalization to [-1,1]
        if self.normalize != "none":
            if self.normalize == "length":
                # normalize such that the largest vector length is 1
                qMax = np.quantile(np.linalg.norm(data, axis=-1), self.normQuantile)
                data = data / qMax

            else:
                if self.normalize == "channel":
                    # normalize each channel individually
                    qMin = np.quantile(data, 1-self.normQuantile, axis=[0,1,2,3])
                    qMax = np.quantile(data,   self.normQuantile, axis=[0,1,2,3])
                else:
                    # normalize all channels with the single min and max
                    qMin = np.repeat(np.quantile(data, 1-self.normQuantile), 3)
                    qMax = np.repeat(np.quantile(data,   self.normQuantile), 3)

                if (qMin == qMax).any():
                    for i in range(qMin.shape[0]):
                        if qMin[i] == qMax[i]:
                            data[..., i] = data[..., i] - qMin[i]
                        else:
                            temp = (data[..., i] - qMin[i]) / (qMax[i] - qMin[i])
                            data[..., i] = (self.normMax - self.normMin) * temp + self.normMin
                else:
                    data = (self.normMax - self.normMin) * ( (data - qMin) / (qMax - qMin) ) + self.normMin

        # reduce number of pairs with index cutoff
        if self.cutoff > 0:
            indexA = indexA[0:self.cutoff]
            indexB = indexB[0:self.cutoff]
            distance = distance[0:self.cutoff]

        # random index permutation
        if self.permuteIndex:
            perm = self.randGen.permutation(len(indexA))
            indexA = indexA[perm]
            indexB = indexB[perm]
            distance = distance[perm]

        # repeat for scalar fields
        if data.shape[data.ndim-1] == 1:
            data = np.repeat(data, 3, axis=data.ndim-1)
        # a data shape of 11x128x128x128x3 from here on

        # random flip
        if self.flip:
            rand = self.randGen.rand(3) > 0.5
            for i in range(len(rand)):
                if rand[i]:
                    data = np.flip(data, axis=i+1)

        # random 90 degree rotation
        if self.rot90:
            angle = self.randGen.randint(0,4,size=3)
            for i in range(len(angle)):
                if angle[i] > 0:
                    data = np.rot90(data, angle[i], axes=self.rotAxes[i])

        # channel swap
        if self.channelSwap:
            channelOrder = [0,1,2]
            self.randGen.shuffle(channelOrder)
            data = data[..., channelOrder]

        # random crop
        if self.crop > 0:
            if not self.cropRandom:
                # to 11xSxSxSx3 where s = output size
                s = [self.crop, self.crop, self.crop]
            else:
                # to 11xSxSxSx3 where s in [output size, 128]
                s = [self.randGen.randint(self.crop, data.shape[1]+1),
                    self.randGen.randint(self.crop, data.shape[2]+1),
                    self.randGen.randint(self.crop, data.shape[3]+1)]

            if (not self.ignoreCrop) and (s[0] < data.shape[1] or s[1] < data.shape[2] or s[2] < data.shape[3]):
                c1 = self.randGen.randint(0, data.shape[1] - s[0]+1)
                c2 = self.randGen.randint(0, data.shape[2] - s[1]+1)
                c3 = self.randGen.randint(0, data.shape[3] - s[2]+1)
                data = data[:, c1:c1+s[0], c2:c2+s[1], c3:c3+s[2]]

        # toTensor
        result = torch.from_numpy(data.copy()).float()

        return {"path": sample["path"], "data": result, "indexA":indexA, "indexB":indexB, "distance":distance}



# combines normalization, repeat (only for scalar fields), resize, to tensor
class TransformsInference(object):
    def __init__(self, normalize:str, order:int, params:Params):
        self.normalize = normalize
        self.outputSize = params.dataScaleInference
        self.order = order
        self.normQuantile = params.dataNormQuant
        self.normMin = params.dataNormMin
        self.normMax = params.dataNormMax

        assert self.normalize in ["none", "length", "channel", "single"]

    def __call__(self, sample:dict):
        data = sample["data"]

        # quantile normalization to [-1,1]
        if self.normalize != "none":
            if self.normalize == "length":
                # normalize such that the largest vector length is 1
                qMax = np.quantile(np.linalg.norm(data, axis=-1), self.normQuantile)
                data = data / qMax

            else:
                if self.normalize == "channel":
                    # normalize each channel individually
                    qMin = np.quantile(data, 1-self.normQuantile, axis=[0,1,2,3])
                    qMax = np.quantile(data,   self.normQuantile, axis=[0,1,2,3])
                else:
                    # normalize all channels with the single min and max
                    qMin = np.repeat(np.quantile(data, 1-self.normQuantile), 3)
                    qMax = np.repeat(np.quantile(data,   self.normQuantile), 3)

                if (qMin == qMax).any():
                    for i in range(qMin.shape[0]):
                        if qMin[i] == qMax[i]:
                            data[..., i] = data[..., i] - qMin[i]
                        else:
                            temp = (data[..., i] - qMin[i]) / (qMax[i] - qMin[i])
                            data[..., i] = (self.normMax - self.normMin) * temp + self.normMin
                else:
                    data = (self.normMax - self.normMin) * ( (data - qMin) / (qMax - qMin) ) + self.normMin

        # repeat for scalar fields
        if data.shape[data.ndim-1] == 1:
            data = np.repeat(data, 3, axis=data.ndim-1)
        # a data shape of 11x128x128x128x3 from here on

        # resize to 11xSxSxSx3
        s = self.outputSize
        if self.outputSize > 0:
            if (s != data.shape[1] or s != data.shape[2] or s != data.shape[3]):
                zoom = [1, s/data.shape[1], s/data.shape[2], s/data.shape[3], 1]
                print("Resize inference transform: " + str(data.shape) + " , scale factor:" + str(zoom))
                data = scipy.ndimage.zoom(data, zoom, order=self.order)

        # toTensor
        result = torch.from_numpy(data).float()

        return {"path": sample["path"], "data": result, "indexA":sample["indexA"], "indexB":sample["indexB"], "distance":sample["distance"]}

