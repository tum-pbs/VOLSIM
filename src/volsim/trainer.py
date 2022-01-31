import logging
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from volsim.base_models import *
from volsim.distance_model import *
from volsim.metrics import *
from volsim.loss import *
from volsim.params import *




# helper method for trainer, validator and tester
def predict(sample, model:nn.Module, i:int, sampleSlicing:int, slicingCount:int):
    sample["idxMin"] = i
    sample["idxMax"] = min(i + sampleSlicing, slicingCount)

    target = sample["distance"][:,sample["idxMin"]:sample["idxMax"]]
    path = np.array(sample["path"])

    # predict
    prediction = model(sample)

    return target, prediction, path





class Trainer(object):
    def __init__(self, model:nn.Module, trainLoader:DataLoader, optimizer:Optimizer, criterion:CorrelationLoss,
                writer:SummaryWriter, params:Params, printEvery:int, logSampleCorrelation:bool=False, showProgressionPrint:bool=False):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.hp = params
        self.printEvery = printEvery
        self.logSampleCorrelation = logSampleCorrelation
        self.showProgressionPrint = showProgressionPrint

        self.history = []

    # run one epoch of training
    def trainingStep(self, epoch:int):
        start = time.time()

        for s, sample in enumerate(self.trainLoader, 0):
            # update loss accumulators
            if not self.hp.lossOnlineMean:
                with torch.no_grad():
                    slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
                    for i in range(0, slicingCount, self.hp.sampleSlicing):
                        target, prediction, _ = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                        self.criterion.updateMeanAccs(prediction, target)

            self.optimizer.zero_grad()

            step = len(self.trainLoader) * epoch + s
            pred = []
            targ = []
            # step
            slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
            for i in range(0, slicingCount, self.hp.sampleSlicing):
                target, prediction, path = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                for j in range(target.shape[1]):
                    pred += [prediction[0,j].item()]
                    targ += [target[0,j].item()]

                loss = self.criterion(prediction, target, path)
                loss.backward()  # accumulates gradient for every simulation pair

                if self.logSampleCorrelation:
                    self.writer.add_scalar("train/Sample_Correlation", self.criterion.lastSampleSliceCorr, 100*step + int(i / self.hp.sampleSlicing))

            if self.hp.gradClip > 0:
                clip_grad_norm_(self.model.parameters(), self.hp.gradClip)
            self.optimizer.step()
            h = self.criterion.getStepHistory()
            self.history += [h]

            self.writer.add_scalar("train/Batch_Loss", h[0], step)
            if h[1] > 0:
                self.writer.add_scalar("train/Batch_LossL2", h[1], step)
            if h[2] > 0:
                self.writer.add_scalar("train/Batch_LossCorr", h[2], step)
            if h[4] > 0:
                self.writer.add_scalar("train/Batch_LossRelL2", h[4], step)
            if h[5] > 0:
                self.writer.add_scalar("train/Batch_LossSizeReg", h[5], step)
            self.writer.add_scalar("train/Batch_Correlation", h[3], step)

            if s%1 == 0 and self.showProgressionPrint:
                sampleShape = sample["data"].shape[0] if type(sample) is dict else sample[0]["data"].shape[0]
                print("Training Epoch:", round((100*s*sampleShape)/len(self.trainLoader), 1), "%", end="\r")

            if s % self.printEvery == self.printEvery - 1:
                end = time.time()

                predString = ""
                targString = ""
                for i in range(10):
                    predString += "%0.2f " % pred[i]
                    targString += "%0.2f " % targ[i]

                samplePath = sample["path"][0] if type(sample) is dict else sample[0]["path"][0]
                print('%2.1f min [%2d, %3d] batch loss: %.8f (%.3f + %.3f) cor: %.3f \t%s \t%s-- %s' % (
                    (end-start)/60.0, epoch+1, s + 1, h[0], h[1], h[2], h[3], samplePath, predString, targString))
                logging.info('[%2d, %3d] batch loss: %.5f (%.3f + %.3f) cor: %.3f \t%s \t%s-- %s' % (
                    epoch+1, s + 1, h[0], h[1], h[2], h[3], samplePath, predString, targString))

        if self.showProgressionPrint:
            print("                                                          ", end="\r") #override temporary percentage print

        hMean = np.mean(np.stack(self.history), axis=0)
        hStd = np.std(np.stack(self.history), axis=0)
        hLow = hMean - hStd
        hHigh = hMean + hStd
        self.writer.add_scalar("train/Epoch_Loss", hMean[0], epoch)
        if hMean[1] > 0:
            self.writer.add_scalar("train/Epoch_LossL2", hMean[1], epoch)
        if hMean[2] > 0:
            self.writer.add_scalar("train/Epoch_LossCorr", hMean[2], epoch)
        if hMean[4] > 0:
            self.writer.add_scalar("train/Epoch_LossRelL2", hMean[4], epoch)
        if hMean[5] > 0:
            self.writer.add_scalar("train/Epoch_LossSizeReg", hMean[5], epoch)
        self.writer.add_scalar("train/Epoch_CorrelationMean", hMean[3], epoch)
        self.writer.add_scalar("train/Epoch_CorrelationMeanLow", hLow[3], epoch)
        self.writer.add_scalar("train/Epoch_CorrelationMeanHigh", hHigh[3], epoch)
        corrAll,_ = self.criterion.getEpochHistory()
        self.writer.add_scalar("train/Epoch_CorrelationFull", corrAll, epoch)

        self.history = []


    # preprocessing step to update interal accumulators of the model for feature map normalization
    def normCalibration(self, epochs:int, ignore:list=[], stopEarly:int=0):
        if self.hp.mNormMode == "norm":
            return

        self.model.eval()
        self.trainLoader.dataset.transform.ignoreCrop = True

        with torch.no_grad():
            for epoch in range(epochs):
                for s, sample in enumerate(self.trainLoader, 0):
                    if stopEarly > 0 and s >= stopEarly:
                        break
                    if any( item in sample["path"][0] for item in ignore ):
                        continue

                    if s%4 == 0 and self.showProgressionPrint:
                        print("Norm calibration:", round((100*(s+epoch*len(self.trainLoader)))/(epochs*len(self.trainLoader)), 1), "%", end="\r")
                    self.model.updateNorm(sample)

        self.trainLoader.dataset.transform.ignoreCrop = False
        self.model.train()
        print("Norm calibration: completed")




class Validator(object):
    def __init__(self, model:nn.Module, valLoader:DataLoader, criterion:CorrelationLoss, writer:SummaryWriter, params:Params):
        self.model = model
        self.valLoader = valLoader
        self.criterion = criterion
        self.writer = writer
        self.hp = params

        self.history = []

        self.finalCorr = {}

    # run one epoch of validation
    def validationStep(self, epoch:int, split:dict, final:bool=False, stepInterval:int=1):
        start = time.time()
        self.model.eval()

        with torch.no_grad():

            for s, sample in enumerate(self.valLoader, 0):
                # update loss accumulators
                if not type(self.model) is Metric and not self.hp.lossOnlineMean:
                    slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
                    for i in range(0, slicingCount, self.hp.sampleSlicing):
                        target, prediction, _ = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                        self.criterion.updateMeanAccs(prediction, target)

                # step
                slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
                for i in range(0, slicingCount, self.hp.sampleSlicing):
                    target, prediction, path = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                    loss = self.criterion(prediction, target, path)

                self.history += [self.criterion.getStepHistory()]

                h = self.history[-1]
                step = len(self.valLoader) * int(epoch/float(stepInterval)) + s
                self.writer.add_scalar("val/Batch_Correlation", h[3], step)

        self.model.train()
        end = time.time()

        hMean = np.mean(np.stack(self.history), axis=0)
        hStd = np.std(np.stack(self.history), axis=0)
        hLow = hMean - hStd
        hHigh = hMean + hStd
        self.writer.add_scalar("val/Epoch_Distance", hMean[1], epoch)
        self.writer.add_scalar("val/Epoch_DistanceLow", hLow[1], epoch)
        self.writer.add_scalar("val/Epoch_DistanceHigh", hHigh[1], epoch)
        self.writer.add_scalar("val/Epoch_CorrelationMean", hMean[3], epoch)
        self.writer.add_scalar("val/Epoch_CorrelationMeanLow", hLow[3], epoch)
        self.writer.add_scalar("val/Epoch_CorrelationMeanHigh", hHigh[3], epoch)

        corrAll, corrSplit = self.criterion.getEpochHistory(split)

        self.writer.add_scalar("val/Epoch_CorrelationFull", corrAll, epoch)
        self.writer.add_scalar("datasets/Correlation_ValAll", corrAll, epoch)
        for split in corrSplit:
            self.writer.add_scalar("datasets/Correlation_" + split, corrSplit[split], epoch)

        if final:
            self.finalCorr = corrSplit
            self.finalCorr["AllVal"] = corrAll

        self.history = []

        print("%3ds Validation: dist error mean (std), corr mean(std), corr full:" % (end-start))
        print("%1.3f (%1.3f) -- %1.3f (%1.3f) -- %1.3f" % (hMean[1], hStd[1], hMean[3], hStd[3], corrAll) )
        print("")
        logging.info("%3ds Validation: dist error mean (std), corr mean(std), corr full:" % (end-start))
        logging.info("%1.3f (%1.3f) -- %1.3f (%1.3f) -- %1.3f" % (hMean[1], hStd[1], hMean[3], hStd[3], corrAll) )
        logging.info("")



class Tester(object):
    def __init__(self, model:nn.Module, testLoader:DataLoader, criterion:CorrelationLoss, writer:SummaryWriter, params:Params):
        self.model = model
        self.testLoader = testLoader
        self.criterion = criterion
        self.writer = writer
        self.hp = params

        self.history = []

        self.finalCorr = {}

    # run one epoch of testing
    def testStep(self, epoch:int, split:dict, final:bool=False, stepInterval:int=1):
        start = time.time()
        self.model.eval()

        with torch.no_grad():

            for s, sample in enumerate(self.testLoader, 0):
                # update loss accumulators
                if not type(self.model) is Metric and not self.hp.lossOnlineMean:
                    slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
                    for i in range(0, slicingCount, self.hp.sampleSlicing):
                        target, prediction, _ = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                        self.criterion.updateMeanAccs(prediction, target)

                # step
                slicingCount = sample["indexA"].shape[1] if type(sample) is dict else sample[0]["indexA"].shape[1]
                for i in range(0, slicingCount, self.hp.sampleSlicing):
                    target, prediction, path = predict(sample, self.model, i, self.hp.sampleSlicing, slicingCount)
                    loss = self.criterion(prediction, target, path)

                self.history += [self.criterion.getStepHistory()]

                h = self.history[-1]
                step = len(self.testLoader) * int(epoch/float(stepInterval)) + s
                self.writer.add_scalar("test/Batch_Correlation", h[3], step)

        self.model.train()
        end = time.time()

        hMean = np.mean(np.stack(self.history), axis=0)
        hStd = np.std(np.stack(self.history), axis=0)
        hLow = hMean - hStd
        hHigh = hMean + hStd
        self.writer.add_scalar("test/Epoch_Distance", hMean[1], epoch)
        self.writer.add_scalar("test/Epoch_DistanceLow", hLow[1], epoch)
        self.writer.add_scalar("test/Epoch_DistanceHigh", hHigh[1], epoch)
        self.writer.add_scalar("test/Epoch_CorrelationMean", hMean[3], epoch)
        self.writer.add_scalar("test/Epoch_CorrelationMeanLow", hLow[3], epoch)
        self.writer.add_scalar("test/Epoch_CorrelationMeanHigh", hHigh[3], epoch)

        corrAll, corrSplit = self.criterion.getEpochHistory(split)

        self.writer.add_scalar("test/Epoch_CorrelationFull", corrAll, epoch)
        self.writer.add_scalar("datasets/Correlation_TestAll", corrAll, epoch)
        for split in corrSplit:
            self.writer.add_scalar("datasets/Correlation_" + split, corrSplit[split], epoch)

        if final:
            self.finalCorr = corrSplit
            self.finalCorr["AllTest"] = corrAll

        self.history = []

        print("%3ds Test: dist error mean (std), corr mean(std), corr full:" % (end-start))
        print("%1.3f (%1.3f) -- %1.3f (%1.3f) -- %1.3f" % (hMean[1], hStd[1], hMean[3], hStd[3], corrAll) )
        print("")
        logging.info("%3ds Test: dist error mean (std), corr mean(std), corr full:" % (end-start))
        logging.info("%1.3f (%1.3f) -- %1.3f (%1.3f) -- %1.3f" % (hMean[1], hStd[1], hMean[3], hStd[3], corrAll) )
        logging.info("")


    def writeFinalAccuracy(self, hParams:Params, finalValAcc:dict):
        print("")
        print("Final validation and test correlation:")
        logging.info("")
        logging.info("Final validation and test correlation:")

        accuracy = {}
        for acc in finalValAcc:
            accuracy["metrics/" + acc] = finalValAcc[acc]
            print("%s: %1.3f" % (acc, finalValAcc[acc]))
            logging.info("%s: %1.3f" % (acc, finalValAcc[acc]))

        for acc in self.finalCorr:
            assert not "metrics/" + acc in accuracy, "Duplicate accuracy keys!"
            accuracy["metrics/" + acc] = self.finalCorr[acc]
            print("%s: %1.3f" % (acc, self.finalCorr[acc]))
            logging.info("%s: %1.3f" % (acc, self.finalCorr[acc]))

        self.writer.add_hparams(hParams.asDict(), accuracy)




