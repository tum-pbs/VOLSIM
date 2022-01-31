import numpy as np
import scipy.stats.stats as sciStats
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


from volsim.params import *

class CorrelationLoss(nn.modules.loss._Loss):
    def __init__(self, params:Params, useGPU:bool):
        super(CorrelationLoss, self).__init__()
        self.useGPU = useGPU
        if useGPU:
            self.epsilon = torch.tensor(0.0000001).cuda()
        else:
            self.epsilon = torch.tensor(0.0000001)
        self.params = params
        self.corHistoryMode = params.corHistoryMode
        self.weightMSE = params.lossFacMSE
        self.weightRelMSE = params.lossFacRelMSE
        self.weightPearsonCorr = params.lossFacPearsonCorr
        self.weightSlConvReg = params.lossFacSlConvReg
        self.weightSizeReg = params.lossFacSizeReg
        self.sizeRegExp = params.lossSizeExp
        self.useOnlineMean = params.lossOnlineMean
        self.aggregateCorr = params.lossCorrAggregate

        self.resetCorrAcc()

        self.stepHist = np.zeros(6)
        self.stepHistCount = 0

        self.lastSampleSliceCorr = 0
        self.epochHist = {"pred":[], "targ":[], "path":[], "enstd":[], "tempPred":[], "tempTarg":[], "tempPath":[], "tempEnstd":[]}

    # has to be called after all simulation pairs of one sample are processed
    # to ensure correct loss computation for next sample 
    def resetCorrAcc(self):
        self.accX = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        self.accY = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        self.count = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        self.accFinal = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        self.countFinal = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        self.accX.requires_grad = False
        self.accY.requires_grad = False
        self.count.requires_grad = False
        self.accFinal.requires_grad = False
        self.countFinal.requires_grad = False


    def forward(self, prediction:torch.Tensor, target:torch.Tensor, path:np.ndarray) -> torch.Tensor:
        if self.useGPU:
            prediction = prediction.cuda()
            target = target.cuda()

        corr = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        correlation = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        # pearson correlation
        if self.weightPearsonCorr > 0:
            corr = self.pearsonCorrOnline(prediction, target)
            self.lastSampleSliceCorr = torch.mean(corr).item()
            correlation = self.weightPearsonCorr * 0.5 * (1-corr)

        # mse
        l2 = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        if self.weightMSE > 0:
            l2 = self.weightMSE * self.distanceL2(prediction, target)

        # relative mse
        relL2 = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        if self.weightRelMSE > 0:
            predMean = self.accX.detach() / self.count.detach()
            targMean = self.accY.detach() / self.count.detach()
            relL2 = self.weightRelMSE * self.distanceL2(prediction-predMean, target-targMean)

        # size regularization
        sizeReg = torch.tensor([0.0]).cuda() if self.useGPU else torch.tensor([0.0])
        if self.weightSizeReg > 0:
            temp = torch.where(prediction > 1, torch.pow(prediction-1, self.sizeRegExp), torch.zeros_like(prediction))
            sizeReg = self.weightSizeReg * torch.sum(temp, dim=1)

        # step history
        self.stepHist = self.stepHist + np.array([
            torch.mean(l2+relL2+correlation+sizeReg).item(),
            torch.mean(l2).item(),
            torch.mean(correlation).item(),
            torch.mean(corr).item(),
            torch.mean(relL2).item(),
            torch.mean(sizeReg).item(),
        ])
        self.stepHistCount = self.stepHistCount + 1

        # epoch history
        self.epochHist["tempPred"] += [prediction.cpu().detach().numpy()]
        self.epochHist["tempTarg"] += [target.cpu().detach().numpy()]
        self.epochHist["tempPath"] += [np.repeat(path[:,None], target.shape[1], axis=1)]

        result = torch.mean(l2 + relL2 + correlation + sizeReg)
        if torch.isnan(result):
            logging.error("NAN in loss!")
            logging.error("L2 " + str(l2))
            logging.error("Rel L2 " + str(relL2))
            logging.error("Corr " + str(corr))
            logging.error("Correlation " + str(correlation))
            raise ValueError("NAN in loss!")
        return result


    def updateMeanAccs(self, x:torch.Tensor, y:torch.Tensor):
        if self.useGPU:
            x = x.cuda()
            y = y.cuda()

        self.count = self.count + x.shape[1]
        self.accX = self.accX + torch.sum(x, dim=1, keepdim=True)
        self.accY = self.accY + torch.sum(y, dim=1, keepdim=True)


    def pearsonCorrOnline(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        if self.useOnlineMean:
            self.updateMeanAccs(x, y)

        if self.count <= 1:
            return torch.zeros_like(self.accFinal)

        meanX = self.accX.detach() / self.count.detach()
        meanY = self.accY.detach() / self.count.detach()
        xm = x - meanX
        ym = y - meanY
        rNum = torch.sum(xm*ym, dim=1, keepdim=True) #manual dot product
        rDen = torch.norm(xm, 2, dim=1, keepdim=True) * torch.norm(ym, 2, dim=1, keepdim=True)
        rVal = rNum / torch.max(rDen, self.epsilon) #epsilon for numerical stability

        if any(torch.isnan(rVal)):
            logging.error("NAN in correlation computation!")
            logging.error("x " + str(x))
            logging.error("y " + str(y))
            logging.error("accX " + str(self.accX))
            logging.error("accY " + str(self.accY))
            logging.error("count " + str(self.count))
            logging.error("meanX " + str(meanX))
            logging.error("meanY " + str(meanY))
            logging.error("rNum " + str(rNum))
            logging.error("rDen " + str(rDen))
            logging.error("rVal " + str(rVal))
            raise ValueError("NAN in correlation computation!")

        if self.aggregateCorr:
            # average over previous pairs from same sample for better stability
            self.accFinal = self.accFinal.detach() + rVal
            self.countFinal = self.countFinal.detach() + 1
            return self.accFinal / self.countFinal
        else:
            return rVal


    def getStepHistory(self) -> np.ndarray:
        result = self.stepHist / self.stepHistCount
        self.stepHist = np.zeros(6)
        self.stepHistCount = 0
        self.resetCorrAcc()

        # normalize all step distances to [0.1, 1.0]
        predStep = np.concatenate(self.epochHist["tempPred"], axis=1) #[3,55]
        dMax = np.max(predStep, axis=1, keepdims=True) #[3,1]
        dMin = np.min(predStep, axis=1, keepdims=True) #[3,1]
        if (dMin == dMax).all():
            predStep = predStep - dMin + 0.1
        elif (dMin == dMax).any():
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    predStep[i] = predStep[i] - dMin[i] + 0.1
                else:
                    predStep[i] = 0.9 * ((predStep[i] - dMin[i]) / (dMax[i] - dMin[i])) + 0.1
        else:
            predStep = 0.9 * ((predStep - dMin) / (dMax - dMin)) + 0.1

        self.epochHist["pred"] += [predStep]
        self.epochHist["targ"] += [np.concatenate(self.epochHist["tempTarg"], axis=1)]
        self.epochHist["path"] += [np.concatenate(self.epochHist["tempPath"], axis=1)]
        self.epochHist["tempPred"] = []
        self.epochHist["tempTarg"] = []
        self.epochHist["tempPath"] = []
        return result

    def getEpochHistory(self, splits:dict=None) -> tuple:
        predEpoch = np.concatenate(self.epochHist["pred"], axis=0)
        targEpoch = np.concatenate(self.epochHist["targ"], axis=0)
        pathEpoch = np.concatenate(self.epochHist["path"], axis=0)

        corrSplit = {}
        if splits:
            for split in splits:
                idx = np.core.defchararray.find(pathEpoch.astype(str), splits[split]) >= 0
                stacked = np.stack([predEpoch[idx], targEpoch[idx]])
                if self.corHistoryMode == "pearson":
                    corr = np.corrcoef(stacked)[0,1]
                elif self.corHistoryMode == "spearman":
                    corr, _ = sciStats.spearmanr(stacked.transpose((1,0)))
                else:
                    raise ValueError("Invalid ground ")
                corrSplit[split] = corr

        stackedAll = np.stack([predEpoch.flatten(), targEpoch.flatten()])
        if self.corHistoryMode == "pearson":
            corrAll = np.corrcoef(stackedAll)[0,1]
        elif self.corHistoryMode == "spearman":
            corrAll, _ = sciStats.spearmanr(stackedAll.transpose((1,0)))
        else:
            raise ValueError("Invalid ground ")

        self.epochHist["pred"] = []
        self.epochHist["targ"] = []
        self.epochHist["path"] = []
        return corrAll, corrSplit

    def distanceL2(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y, reduction='none')

    def distanceL1(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return F.l1_loss(x, y, reduction='none')
