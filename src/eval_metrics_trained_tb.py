import os
from torch.utils.data import DataLoader

from volsim.logger import *
from volsim.params import *
from volsim.simulation_dataset import *
from volsim.distance_model import *
from volsim.loss import *
from volsim.trainer import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.manual_seed(1)

useGPU = True
volsim = DistanceModel.load("models/VolSiM.pth", useGPU=useGPU)
cnnTrained = DistanceModel.load("models/CNN_trained.pth", useGPU=useGPU)
msRand = DistanceModel.load("models/MS_rand.pth", useGPU=useGPU)
cnnRand = DistanceModel.load("models/CNN_rand.pth", useGPU=useGPU)
msIdentity = DistanceModel.load("models/MS_identity.pth", useGPU=useGPU)
msNoSkip = DistanceModel.load("models/MS_noSkip.pth", useGPU=useGPU)
ms3Scales = DistanceModel.load("models/MS_3scales.pth", useGPU=useGPU)
ms5Scales = DistanceModel.load("models/MS_5scales.pth", useGPU=useGPU)
msAddedIso = DistanceModel.load("models/MS_addedIso.pth", useGPU=useGPU)
msOnlyIso = DistanceModel.load("models/MS_onlyIso.pth", useGPU=useGPU)
metrics = {"VolSiM":volsim, "CNN_trained":cnnTrained, "MS_rand":msRand, "CNN_rand":cnnRand,
            "MS_identity":msIdentity, "MS_noSkip":msNoSkip, "MS_3scales":ms3Scales, "MS_5scales":ms5Scales,
            "MS_addedIso":msAddedIso, "MS_onlyIso":msOnlyIso}

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

valSet.printDatasetInfo()
testSet.printDatasetInfo()
valSet.loadDistanceCoefficients("lin")
testSet.loadDistanceCoefficients("lin")


for name in metrics.keys():
    print("\n"+name)
    model = metrics[name]
    hp = model.hp
    logger = Logger("trained/%d_%s" % (dataSize, name), params=hp, override=False, addNumber=True)

    transVal = TransformsInference(normalize="single", order=0, params=hp)
    valSet.setDataTransform(transVal)
    testSet.setDataTransform(transVal)

    valLoader = DataLoader(valSet, batch_size=hp.batch, shuffle=True, num_workers=4)
    testLoader = DataLoader(testSet, batch_size=hp.batch, shuffle=True, num_workers=4)

    criterion = CorrelationLoss(params=hp, useGPU=useGPU)
    validator = Validator(model, valLoader, criterion, logger.tfWriter, hp)
    tester = Tester(model, testLoader, criterion, logger.tfWriter, hp)

    logger.setup(model, None, None, valSet.split, testSet.split)

    validator.validationStep(0, valSet.split, final=True)
    tester.testStep(0, testSet.split, final=True)

    tester.writeFinalAccuracy(hp, validator.finalCorr)

    logger.close()
