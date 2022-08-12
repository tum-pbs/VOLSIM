import os
import torch
from torch.utils.data import DataLoader

from volsim.logger import *
from volsim.params import *
from volsim.simulation_dataset import *
from volsim.distance_model import *
from volsim.loss import *
from volsim.trainer import *
from volsim.lr_policies import *

if __name__ == '__main__':
    useGPU = True
    gpuID = "0"

    #torch.manual_seed(1)

    modelName = "VolSiM"
    hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
                gtDistMode="fit2", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
                lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
                dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="multiScaleSkip_16_1",
                mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")

    #modelName = "CNN_trained"
    #hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
    #            gtDistMode="fit2", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
    #            lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
    #            dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="alex_5_1",
    #            mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")

    #modelName = "MS_identity"
    #hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
    #            gtDistMode="lin", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
    #            lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
    #            dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="multiScaleSkip_16_1",
    #            mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")

    #modelName = "MS_noSkip"
    #hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
    #            gtDistMode="fit2", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
    #            lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
    #            dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="multiScale_16_1",
    #            mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")

    #modelName = "MS_3scales"
    #hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
    #            gtDistMode="fit2", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
    #            lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
    #            dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="multiScaleSkip_12_1",
    #            mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")

    #modelName = "MS_5scales"
    #hp = Params(batch=1, epochs=40, lrBase=0.00010, lrDecFac=1.0, lrDecTimes=1.0, weightDecay=0.0,
    #            gtDistMode="fit2", corHistoryMode="spearman", lossFacPearsonCorr=0.7, lossFacMSE=1.0,
    #            lossOnlineMean=True, lossCorrAggregate=True, sampleSlicing=55, dataAugmentation=True, dataCrop=64,
    #            dataCropRandom=False, dataScaleInference=64, dataConvertMode="none", mBase="multiScaleSkip_20_1",
    #            mFeatDist="L2Sqrt", mLinInit=0.05, mLinDropout=True, mNormMode="normMeanLayerGlobal")


    dataSize = 64
    trainSet = SimulationDataset("Training", ["data/%d_train" % (dataSize)], dataMode="full",
                filterSimDict={"smoke" : ["000000"], "liquid" : ["000000"],
                            "advdiff":["000000", "000001", "000002"],
                            "burgers":["000000", "000001", "000002"]},
                excludeFilterSim=True,
                #overfitCount=2,
                printLevel="sim")

    valSet = SimulationDataset("Validation", ["data/%d_train" % (dataSize)], dataMode="full",
                filterSimDict={"smoke" : ["000000"], "liquid" : ["000000"],
                            "advdiff":["000000", "000001", "000002"],
                            "burgers":["000000", "000001", "000002"]},
                split = {"Adv_" : "advdiff", "Bur_" : "burgers", "Liq_" : "liquid", "Smo_" : "smoke"},
                #overfitCount=2,
                printLevel="sim")

    testSet = SimulationDataset("Test", ["data/%d_test" % (dataSize)], dataMode="full",
                split = {"AdvD" : "advdiff_dens", "LiqN" : "liquid_bgnoise", "JHTDB_iso" : "jhtdb_isotropic1024coarse", 
                        "JHTDB_cha" : "jhtdb_channel", "JHTDB_mhd" : "jhtdb_mhd1024", "JHTDB_rot" : "jhtdb_rotstrat4096",
                        "JHTDB_tra" : "jhtdb_transition_bl", "SF" : "scalarflow_step13", "Sha" : "shapes", "Wav" : "waves"},
                #overfitCount=2,
                printLevel="sim")



def train(modelName:str, trainSet:SimulationDataset, valSet:SimulationDataset, testSet:SimulationDataset,
            hp:Params, useGPU:bool=True, gpuID:str="0"):

    # SETUP FOR DATA AND MODEL
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    logger = Logger(modelName, hp, addNumber=True, addDate=False)

    transTrain = TransformsTrain(permuteIndex=True, normalize="single", params=hp)
    transVal = TransformsInference(normalize="single", order=0, params=hp)
    trainSet.setDataTransform(transTrain)
    valSet.setDataTransform(transVal)
    testSet.setDataTransform(transVal)
    trainSet.loadDistanceCoefficients(hp.gtDistMode)
    valSet.loadDistanceCoefficients("lin")
    testSet.loadDistanceCoefficients("lin")
    trainSet.printDatasetInfo()
    valSet.printDatasetInfo()
    testSet.printDatasetInfo()

    trainLoader = DataLoader(trainSet, batch_size=hp.batch, shuffle=True, num_workers=3)
    valLoader = DataLoader(valSet, batch_size=hp.batch, shuffle=True, num_workers=3)
    testLoader = DataLoader(testSet, batch_size=hp.batch, shuffle=True, num_workers=3)

    model = DistanceModel(modelParams=hp, useGPU=useGPU)
    model.printModelInfo()

    criterion = CorrelationLoss(params=hp, useGPU=useGPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lrBase, weight_decay=hp.weightDecay)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LrDecayPolicy(hp.lrDecFac, hp.lrDecTimes, hp.epochs))
    trainer = Trainer(model, trainLoader, optimizer, criterion, logger.tfWriter, hp, 1, False, False)
    validator = Validator(model, valLoader, criterion, logger.tfWriter, hp)
    tester = Tester(model, testLoader, criterion, logger.tfWriter, hp)

    logger.setup(model, optimizer, lrScheduler, valSet.split, testSet.split)

    # ACTUAL TRAINING
    print('Starting Training')

    if hp.mNormMode != "normUnit":
        trainer.normCalibration(hp.calibEpochs, hp.calibIgnore, stopEarly=0)

    validator.validationStep(0, valSet.split, stepInterval=1)
    tester.testStep(0, testSet.split, stepInterval=1)

    logger.saveTrainState(0, milestone=True)
    logger.saveTrainState(0)

    for epoch in range(0, hp.epochs):
        trainer.trainingStep(epoch)
        lrScheduler.step()
        logger.saveTrainState(epoch)

        if epoch != hp.epochs-1:
            validator.validationStep(epoch, valSet.split, stepInterval=1)
            tester.testStep(epoch, testSet.split, stepInterval=1)
            logger.saveTrainState(epoch, milestone=True)

    validator.validationStep(hp.epochs, valSet.split, final=True, stepInterval=1)
    tester.testStep(hp.epochs, testSet.split, final=True, stepInterval=1)

    print('Finished Training')
    tester.writeFinalAccuracy(hp, validator.finalCorr)
    logger.close()



if __name__ == '__main__':
    train(modelName, trainSet, valSet, testSet, hp, useGPU=useGPU, gpuID=gpuID) #type:ignore