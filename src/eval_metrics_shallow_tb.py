from torch.utils.data import DataLoader

from volsim.logger import *
from volsim.params import *
from volsim.simulation_dataset import *
from volsim.metrics import *
from volsim.loss import *
from volsim.trainer import *


#torch.manual_seed(1)

metrics = [Metric("MSE"), Metric("SSIM"), Metric("PSNR"), Metric("MI"), Metric("LPIPS"), Metric("LSIM2D"), ]

dataSize = 64
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

transVal = TransformsInference(normalize="single", order=0, params=Params(dataScaleInference=dataSize, dataConvertMode="none", dataNormQuant=1.0, dataNormMin=0.0, dataNormMax=255.0))
valSet.setDataTransform(transVal)
testSet.setDataTransform(transVal)
valSet.loadDistanceCoefficients("lin")
testSet.loadDistanceCoefficients("lin")
valSet.printDatasetInfo()
testSet.printDatasetInfo()

batch = 1
valLoader = DataLoader(valSet, batch_size=batch, shuffle=True, num_workers=4)
testLoader = DataLoader(testSet, batch_size=batch, shuffle=True, num_workers=4)

for metric in metrics:
    print("\n"+metric.name)
    hp = Params(mBase=metric.mode, batch=batch, epochs=0, lrBase=0, lrAvg=0, lrDecFac=0, lrDecTimes=0, weightDecay=0, gradClip=0,
                lossFacMSE=1, lossFacPearsonCorr=1, lossOnlineMean=False, lossCorrAggregate=False, sampleSlicing=55,
                calibEpochs=0, dataAugmentation=False, dataNormQuant=1.0, dataNormMin=0.0, dataNormMax=255.0,
                dataCrop=0, dataScaleInference=dataSize, mLinInit=0, mLinDropout=False, mBaseInit="-",
                mFeatDist="-", mNormMode="-")
    metric.hp = hp

    logger = Logger("shallow/%d_%s" % (dataSize, metric.name), params=None, override=False, addNumber=True)

    criterion = CorrelationLoss(params=hp, useGPU=False)
    validator = Validator(metric, valLoader, criterion, logger.tfWriter, hp)
    tester = Tester(metric, testLoader, criterion, logger.tfWriter, hp)

    logger.setup(metric, None, None, valSet.split, testSet.split)

    validator.validationStep(hp.epochs, valSet.split, final=True)
    tester.testStep(hp.epochs, testSet.split, final=True)

    tester.writeFinalAccuracy(hp, validator.finalCorr)

    logger.close()