import os, logging, math
import numpy as np
import torch
import torch.nn as nn

from volsim.base_models import *
from volsim.simulation_dataset import *
from volsim.params import *


class DistanceModel(nn.Module):

    def __init__(self, modelParams:Params, useGPU:bool=True):
        super(DistanceModel, self).__init__()

        self.hp = modelParams
        self.useGPU = useGPU

        if "multiScale" in self.hp.mBase:
            base = self.hp.mBase.split("_")
            try:
                layers = int(base[1])
            except ValueError:
                layers = 12
            try:
                width = float(base[2])
            except ValueError:
                width = 1
            useSkip = "Skip" in self.hp.mBase
            self.basenet = MultiScaleNet(widthFactor=width, layers=layers, firstChannels=3, useSkip=useSkip)
        elif "alex" in self.hp.mBase:
            base = self.hp.mBase.split("_")
            try:
                layers = int(base[1])
            except ValueError:
                layers = 5
            try:
                width = float(base[2])
            except ValueError:
                width = 1

            convKernel, maxPoolKernel, firstStride = (12, 4, 4)

            self.basenet = AlexNetLike(widthFactor=width, layers=layers, convKernel=convKernel, maxPoolKernel=maxPoolKernel,
                                    firstStride=firstStride)
        else:
            raise ValueError('Unknown base network type.')

        self.normAcc = []  #for normMode max
        self.normM2 = []   #for normMode mean
        for i in range(self.basenet.layers):
            if self.useGPU:
                self.normAcc += [torch.tensor([0.0], requires_grad=False).cuda()]
                self.normM2  += [torch.tensor([0.0], requires_grad=False).cuda()]
            else:
                self.normAcc += [torch.tensor([0.0], requires_grad=False)]
                self.normM2  += [torch.tensor([0.0], requires_grad=False)]
        self.normCount = [0] * self.basenet.layers #for normMode avg

        self.avgs = []
        self.avg0 = self.avgLayer(self.basenet.channels[0])#, self.basenet.featureMapSize[0])
        self.avgs += [self.avg0]
        if self.basenet.layers > 1:
            self.avg1 = self.avgLayer(self.basenet.channels[1])#, self.basenet.featureMapSize[1])
            self.avgs += [self.avg1]
        if self.basenet.layers > 2:
            self.avg2 = self.avgLayer(self.basenet.channels[2])#, self.basenet.featureMapSize[2])
            self.avgs += [self.avg2]
        if self.basenet.layers > 3:
            self.avg3 = self.avgLayer(self.basenet.channels[3])#, self.basenet.featureMapSize[3])
            self.avgs += [self.avg3]
        if self.basenet.layers > 4:
            self.avg4 = self.avgLayer(self.basenet.channels[4])#, self.basenet.featureMapSize[4])
            self.avgs += [self.avg4]
        if self.basenet.layers > 5:
            self.avg5 = self.avgLayer(self.basenet.channels[5])#, self.basenet.featureMapSize[5])
            self.avgs += [self.avg5]
        if self.basenet.layers > 6:
            self.avg6 = self.avgLayer(self.basenet.channels[6])#, self.basenet.featureMapSize[6])
            self.avgs += [self.avg6]
        if self.basenet.layers > 7:
            self.avg7 = self.avgLayer(self.basenet.channels[7])#, self.basenet.featureMapSize[7])
            self.avgs += [self.avg7]
        if self.basenet.layers > 8:
            self.avg8 = self.avgLayer(self.basenet.channels[8])#, self.basenet.featureMapSize[8])
            self.avgs += [self.avg8]
        if self.basenet.layers > 9:
            self.avg9 = self.avgLayer(self.basenet.channels[9])#, self.basenet.featureMapSize[9])
            self.avgs += [self.avg9]
        if self.basenet.layers > 10:
            self.avg10 = self.avgLayer(self.basenet.channels[10])#, self.basenet.featureMapSize[10])
            self.avgs += [self.avg10]
        if self.basenet.layers > 11:
            self.avg11 = self.avgLayer(self.basenet.channels[11])#, self.basenet.featureMapSize[11])
            self.avgs += [self.avg11]
        if self.basenet.layers > 12:
            self.avg12 = self.avgLayer(self.basenet.channels[12])#, self.basenet.featureMapSize[12])
            self.avgs += [self.avg12]
        if self.basenet.layers > 13:
            self.avg13 = self.avgLayer(self.basenet.channels[13])#, self.basenet.featureMapSize[13])
            self.avgs += [self.avg13]
        if self.basenet.layers > 14:
            self.avg14 = self.avgLayer(self.basenet.channels[14])#, self.basenet.featureMapSize[14])
            self.avgs += [self.avg14]
        if self.basenet.layers > 15:
            self.avg15 = self.avgLayer(self.basenet.channels[15])#, self.basenet.featureMapSize[15])
            self.avgs += [self.avg15]
        if self.basenet.layers > 16:
            self.avg16 = self.avgLayer(self.basenet.channels[16])#, self.basenet.featureMapSize[16])
            self.avgs += [self.avg16]
        if self.basenet.layers > 17:
            self.avg17 = self.avgLayer(self.basenet.channels[17])#, self.basenet.featureMapSize[17])
            self.avgs += [self.avg17]
        if self.basenet.layers > 18:
            self.avg18 = self.avgLayer(self.basenet.channels[18])#, self.basenet.featureMapSize[18])
            self.avgs += [self.avg18]
        if self.basenet.layers > 19:
            self.avg19 = self.avgLayer(self.basenet.channels[19])#, self.basenet.featureMapSize[19])
            self.avgs += [self.avg19]

        # initialize learned average weight layers
        for avgLayer in self.avgs:
            for layer in avgLayer:
                if isinstance(layer, nn.Conv3d):
                    layer.weight.data.fill_(self.hp.mLinInit)

        if self.useGPU:
            self.cuda()


    @classmethod
    def load(cls, path:str, useGPU:bool=True):
        if useGPU:
            print('Loading model from %s' % path)
            loaded = torch.load(path)
        else:
            print('CPU - Loading model from %s' % path)
            loaded = torch.load(path, map_location=torch.device('cpu'))

        params = Params.fromDict(loaded['hyperparams'])
        stateDict = loaded['stateDict']
        model = cls(params, useGPU)

        model.load_state_dict(stateDict)
        model.eval()

        if params.mNormMode != "norm":
            model.normAcc = loaded['normAcc']
            model.normM2 = loaded['normM2']
            model.normCount = loaded['normCount']

        return model


    def forward(self, x:dict) -> Tuple[torch.Tensor, list]:
        full = x["data"].cuda() if self.useGPU else x["data"]
        idxA = x["indexA"][0,x["idxMin"]:x["idxMax"]].long() #only use index of first batch element for entire batch
        idxB = x["indexB"][0,x["idxMin"]:x["idxMax"]].long()
        idxA = idxA.cuda() if self.useGPU else idxA
        idxB = idxB.cuda() if self.useGPU else idxB

        dataA = torch.index_select(full, 1, idxA)
        dataB = torch.index_select(full, 1, idxB)

        dataA = dataA.view(-1,full.shape[2],full.shape[3],full.shape[4],full.shape[5])
        dataB = dataB.view(-1,full.shape[2],full.shape[3],full.shape[4],full.shape[5])
        dataA = dataA.permute(0,4,1,2,3) # change shape to [batch*sampleSlice,3,128,128,128]
        dataB = dataB.permute(0,4,1,2,3)

        self.clampWeights()

        outBaseA = self.basenet(dataA)
        outBaseB = self.basenet(dataB)

        result = torch.tensor([[0.0]]).cuda() if self.useGPU else torch.tensor([[0.0]])

        for i in range( len(outBaseA) ):
            if i in self.hp.mIgnoreLayers:
                continue

            #print(outBaseA[i].shape)
            normalized1 = self.normalizeTensor(outBaseA[i], i)
            normalized2 = self.normalizeTensor(outBaseB[i], i)

            if self.hp.mFeatDist == "L1":
                diff = torch.abs(normalized2 - normalized1)
            elif self.hp.mFeatDist == "L2" or self.hp.mFeatDist == "L2Sqrt":
                diff = (normalized2 - normalized1)**2
            else:
                raise ValueError('Unknown feature distance.')

            weightedDiff = self.avgs[i](diff)

            result = result + torch.mean(weightedDiff, dim=[2,3,4])

        if self.hp.mFeatDist == "L2Sqrt":
            result = torch.sqrt(result)

        return torch.squeeze(result, dim=1).view(full.shape[0],-1)


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
        data_transform = TransformsInference("single", 3, self.hp)
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


    # ensures that avg layer weights are greater or equal to zero
    def clampWeights(self):
        for avgLayer in self.avgs:
            for layer in avgLayer:
                if isinstance(layer, nn.Conv3d):
                    layer.weight.data = torch.clamp(layer.weight.data, min=0)


    # 1x1 convolution layer to scale feature maps channel-wise
    def avgLayer(self, channelsIn:int) -> nn.Sequential:
        if self.hp.mLinDropout:
            return nn.Sequential(
                nn.Dropout(),
                nn.Conv3d(channelsIn, 1, 1, stride=1, padding=0, bias=False),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(channelsIn, 1, 1, stride=1, padding=0, bias=False),
            )

    # preprocessing step that updates internal accumulators for feature map normalization
    def updateNorm(self, sample:dict):
        full = sample["data"].cuda() if self.useGPU else sample["data"]
        for i in range(full.shape[1]): # do not use index here, only iterate over all data once
            data = full[:, i]
            data = data.permute(0,4,1,2,3) # change shape to [batch,3,128,128,128]

            self.clampWeights()
            outBase = self.basenet(data)

            for j in range( len(outBase) ):
                self.normalizeTensor(outBase[j], j, updateAcc=True)


    # normalizes feature map tensor along channel dimension with different methods
    def normalizeTensor(self, tensorIn:torch.Tensor, layer:int, epsilon:float=1e-10,
                    updateAcc:bool=False) -> torch.Tensor:
        size = tensorIn.size()

        # unit normalize tensor in channel dimension
        if self.hp.mNormMode == "normUnit":
            norm = torch.sqrt( torch.sum(tensorIn**2,dim=1) )
            norm = norm.view(size[0], 1, size[2], size[3], size[4])
            return tensorIn / (norm.expand_as(tensorIn) + epsilon)

        elif self.hp.mNormMode == "normMeanLayerGlobal":
            if updateAcc:
                self.normCount[layer] = self.normCount[layer] + size[0]
                delta = tensorIn - self.normAcc[layer].expand_as(tensorIn)
                self.normAcc[layer] = self.normAcc[layer] + torch.sum( torch.mean(delta / self.normCount[layer], dim=1) , dim=0)
                self.normM2[layer] = self.normM2[layer] + torch.sum( torch.mean(delta *(tensorIn - self.normAcc[layer].expand_as(tensorIn)), dim=1) , dim=0)

            # rescale norm accumulators for differently sized inputs
            if size[2] != self.normAcc[layer].shape[0] or size[3] != self.normAcc[layer].shape[1] or size[4] != self.normAcc[layer].shape[2]:
                up = nn.Upsample(size=(size[2], size[3], size[4]), mode="trilinear", align_corners=True)
                normAcc = torch.squeeze(up( torch.unsqueeze(torch.unsqueeze(self.normAcc[layer].detach(), dim=0), dim=0) ))
                normM2 = torch.squeeze(up( torch.unsqueeze(torch.unsqueeze(self.normM2[layer].detach(), dim=0), dim=0) ))

                mean = normAcc
                mean = mean.view(1, 1, size[2], size[3], size[4])
                std = torch.sqrt( normM2 / (self.normCount[layer] - 1) )
                std = std.view(1, 1, size[2], size[3], size[4])
            # directly use norm accumulators for matching input size
            else:
                mean = self.normAcc[layer]
                mean = mean.view(1, 1, size[2], size[3], size[4])
                std = torch.sqrt( self.normM2[layer] / (self.normCount[layer] - 1) )
                std = std.view(1, 1, size[2], size[3], size[4])
            normalized = (tensorIn - mean.expand_as(tensorIn)) / (std.expand_as(tensorIn) + epsilon)
            normalized2 = normalized / (math.sqrt(size[1]) - 1)
            return normalized2

        elif self.hp.mNormMode == "normNone":
            return tensorIn
        else:
            raise ValueError('Unknown norm mode.')


    def printModelInfo(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in parameters])
        print("Trainable parameters: %d" % params)
        print(self)
        print("")
        logging.info("Trainable parameters: %d" % params)
        logging.info(self)
        logging.info("")



    def save(self, path:str, override:bool=False, noPrint:bool=False):
        if not noPrint:
            print('Saving model to %s' % path)
        if not override and os.path.isfile(path):
            raise ValueError("Override warning!")
        else:
            saveDict = {'stateDict' : self.state_dict(), 'hyperparams' : self.hp.asDict(),}
            if self.hp.mNormMode != "norm":
                saveDict['normAcc'] = self.normAcc
                saveDict['normM2'] = self.normM2
                saveDict['normCount'] = self.normCount

            torch.save(saveDict, path)


    def resume(self, path:str):
        if self.useGPU:
            print('Resuming model from %s' % path)
            loaded = torch.load(path)
        else:
            print('CPU - Resuming model from %s' % path)
            loaded = torch.load(path, map_location=torch.device('cpu'))

        self.load_state_dict(loaded['stateDict'])
        self.hp = Params().fromDict(loaded['hyperparams'])

        if self.hp.mNormMode != "norm":
            self.normAcc = loaded['normAcc']
            self.normM2 = loaded['normM2']
            self.normCount = loaded['normCount']

