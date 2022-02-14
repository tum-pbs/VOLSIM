import torch
import torch.nn as nn
import numpy as np
import math
import skimage.metrics as metrics
import scipy.ndimage.filters as filters

from volsim.simulation_dataset import *

from lpips.models.dist_model import DistModel as LPIPS_Model
from lsim.distance_model import DistanceModel as LSIM2D_Model


class Metric(nn.Module):
    def __init__(self, mode:str):
        super(Metric, self).__init__()
        assert (mode in ["MSE", "MSE(blurred)", "MSE(fft)", "SSIM", "PSNR", "MI", "CORR", "LPIPS", "LSIM2D"]), "Unknown metric mode!"
        self.mode = mode
        self.name = mode
        self.eval()
        if mode == "LPIPS":
            self.model = LPIPS_Model()
            self.model.initialize(model='net-lin', net='alex', use_gpu=True, spatial=False)
        if mode == "LSIM2D":
            self.model = LSIM2D_Model(baseType="lsim", isTrain=False, useGPU=True)
            self.model.load("src/lsim/models/LSiM.pth")


    def forward(self, x:dict) -> torch.Tensor:
        full = x["data"]
        idxA = x["indexA"][0,x["idxMin"]:x["idxMax"]].long() #only use index of first batch element for entire batch
        idxB = x["indexB"][0,x["idxMin"]:x["idxMax"]].long()

        dataA = torch.index_select(full, 1, idxA)
        dataB = torch.index_select(full, 1, idxB)

        dataA = dataA.view(-1,full.shape[2],full.shape[3],full.shape[4],full.shape[5])
        dataB = dataB.view(-1,full.shape[2],full.shape[3],full.shape[4],full.shape[5])
        dataA = dataA.numpy()
        dataB = dataB.numpy()
        dataAInt = dataA.astype(np.uint8)
        dataBInt = dataB.astype(np.uint8)

        distance = np.empty(dataA.shape[0])
        for i in range(dataA.shape[0]):
            if self.mode == "MSE":
                distance[i] = metrics.mean_squared_error(dataA[i], dataB[i])
            elif self.mode == "MSE(blurred)":
                tempA = filters.gaussian_filter(dataA[i], 2)
                tempB = filters.gaussian_filter(dataB[i], 2)
                distance[i] = metrics.mean_squared_error(tempA, tempB)
            elif self.mode == "MSE(fft)":
                tempA = np.abs(np.fft.fftn(dataA[i]))
                tempB = np.abs(np.fft.fftn(dataB[i]))
                distance[i] = metrics.mean_squared_error(tempA, tempB)
            elif self.mode == "SSIM":
                distance[i] = 1 - metrics.structural_similarity(dataA[i], dataB[i], data_range=255.0, multichannel=True) #invert as distance measure
            elif self.mode == "PSNR":
                psnr = -metrics.peak_signal_noise_ratio(dataA[i], dataB[i], data_range=255) #invert as distance measure
                distance[i] = psnr if not math.isinf(psnr) else -999
            elif self.mode == "MI":
                distance[i] = np.mean(metrics.variation_of_information(dataAInt[i], dataBInt[i]))
            elif self.mode == "CORR":
                tempA = dataA[i].reshape(-1)
                tempB = dataB[i].reshape(-1)
                stacked = np.stack([tempA, tempB], axis=0)
                corr = np.corrcoef(stacked)[0,1]
                if np.isnan(corr):
                    distance[i] = 1 # handle undefined correlation for zero variance
                else:
                    distance[i] = 1 - np.abs(corr) #invert as distance measure

            elif self.mode == "LPIPS":
                minA = np.min(dataA[i])
                maxA = np.max(dataA[i])
                rescaledA = 2 * ( (dataA[i] - minA) / (maxA - minA) ) - 1
                rescaledA = torch.from_numpy(rescaledA).cuda()

                minB = np.min(dataB[i])
                maxB = np.max(dataB[i])
                rescaledB = 2 * ( (dataB[i] - minB) / (maxB - minB) ) - 1
                rescaledB = torch.from_numpy(rescaledB).cuda()

                xPermA = rescaledA.permute(0,3,1,2)
                xPermB = rescaledB.permute(0,3,1,2)
                xResult = self.model(xPermA, xPermB)

                yPermA = rescaledA.permute(1,3,0,2)
                yPermB = rescaledB.permute(1,3,0,2)
                yResult = self.model(yPermA, yPermB)

                zPermA = rescaledA.permute(2,3,0,1)
                zPermB = rescaledB.permute(2,3,0,1)
                zResult = self.model(zPermA, yPermB)
                distance[i] = np.mean( (xResult + yResult + zResult) / 3 )

            elif self.mode == "LSIM2D":
                tensA = torch.from_numpy(dataA[i]).cuda()
                tensB = torch.from_numpy(dataB[i]).cuda()

                xPermA = tensA.permute(0,3,1,2)[None,...]
                xPermB = tensB.permute(0,3,1,2)[None,...]
                xDict = {"reference": xPermA, "other": xPermB}
                xResult = self.model(xDict).cpu().numpy()

                yPermA = tensA.permute(1,3,0,2)[None,...]
                yPermB = tensB.permute(1,3,0,2)[None,...]
                yDict = {"reference": yPermA, "other": yPermB}
                yResult = self.model(yDict).cpu().numpy()

                zPermA = tensA.permute(2,3,0,1)[None,...]
                zPermB = tensB.permute(2,3,0,1)[None,...]
                zDict = {"reference": zPermA, "other": zPermB}
                zResult = self.model(zDict).cpu().numpy()
                distance[i] = np.mean( (xResult + yResult + zResult) / 3 )

        return torch.from_numpy(distance).float().view(full.shape[0], -1)


    # input two numpy arrays with shape [width, height, depth, channel] or shape
    # [batch, width, height, depth, channel] where channel = 1 or channel = 3
    # and return a distance of shape [1] or [batch]
    # If true, normalize performs a normalization to the models native data range jointly for the full data batch
    # If true, interpolate performs a spatial interpolation to the models native data size jointly for the full data batch
    def computeDistance(self, input1:np.ndarray, input2:np.ndarray, normalize:bool, interpolate:bool) -> np.ndarray:
        assert (not self.training), "Distance computation should happen in evaluation mode!"
        assert (input1.shape == input2.shape), "Input shape mismatch!"

        in1 = input1[None,...] if input1.ndim == 4 else input1
        in2 = input2[None,...] if input2.ndim == 4 else input2
        data_transform = TransformsInference("single", 0, Params(dataScaleInference=-1, dataConvertMode="none", dataNormQuant=1.0, dataNormMin=0.0, dataNormMax=255.0))
        if not normalize:
            data_transform.normalize = "none"
        if not interpolate:
            data_transform.outputSize = -1
        data = np.concatenate([in1, in2], axis=0) # stack along param dimension
        dataDict = {"data": data, "path": None, "distance": None, "indexA" : None, "indexB" : None, "idxMin" : None, "idxMax" : None}
        data = data_transform(dataDict)["data"]

        nPairs = in1.shape[0]
        distance = torch.from_numpy(np.zeros(nPairs, dtype=np.float32))
        indexA = torch.from_numpy(np.arange(nPairs, dtype=np.int32))
        indexB = torch.from_numpy(np.arange(nPairs, dtype=np.int32) + nPairs)
        path = np.array([""]*nPairs)

        sample = {"data": data[None,...], "path": path, "distance": distance[None,...],
                "indexA" : indexA[None,...], "indexB" : indexB[None,...], "idxMin" : 0, "idxMax" : nPairs}

        output = self(sample)
        output = output.cpu().detach().view(-1).numpy()
        return output