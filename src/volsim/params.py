import random
import numpy as np

class Params(object):
    def __init__(self, batch=1, epochs=20, lrBase=0.0002, lrAvg=0.00001, lrDecFac=0.5, lrDecTimes=1.0, weightDecay=0.0, gradClip=0,
                gtDistMode="lin", corHistoryMode="spearman", lossFacMSE=1.0, lossFacRelMSE=0.0, lossFacPearsonCorr=0.5,
                lossFacSpearmanCorr=0.0, lossFacSlConvReg=0.0, lossFacSizeReg=0.0, lossSizeExp=2,
                calibEpochs=1, calibIgnore=[], lossOnlineMean=True, lossCorrAggregate=False, sampleSlicing=55,
                dataAugmentation=True, dataNormQuant=1.0, dataNormMin=-1, dataNormMax=1, dataCrop=64, dataCropRandom=False,
                dataScaleInference=64, dataConvertMode="none", dataCutoffIndex=-1, mBase="multiScaleSkip_16_1", mLinInit=0.1,
                mLinDropout=True, mBaseInit="pretrained", mFeatDist="L2Sqrt", mNormMode="normMeanLayerGlobal", mIgnoreLayers=[]):
        self.batch               = batch.eval()               if isinstance(batch               ,PSearch) else batch                # batch size
        self.epochs              = epochs.eval()              if isinstance(epochs              ,PSearch) else epochs               # number of training epochs
        self.lrBase              = lrBase.eval()              if isinstance(lrBase              ,PSearch) else lrBase               # learning rate for base network
        self.lrAvg               = lrAvg.eval()               if isinstance(lrAvg               ,PSearch) else lrAvg                # learning rate for weighted avg layers
        self.lrDecFac            = lrDecFac.eval()            if isinstance(lrDecFac            ,PSearch) else lrDecFac             # learning rate decays smoothly by this factor over all epcohs
        self.lrDecTimes          = lrDecTimes.eval()          if isinstance(lrDecTimes          ,PSearch) else lrDecTimes           # learning rate decays by lrDecFac multiple times during training
        self.weightDecay         = weightDecay.eval()         if isinstance(weightDecay         ,PSearch) else weightDecay          # weight decay factor to regularize the net by penalizing large weights
        self.gradClip            = gradClip.eval()            if isinstance(gradClip            ,PSearch) else gradClip             # clip all gradients larger the this value
        self.gtDistMode          = gtDistMode.eval()          if isinstance(gtDistMode          ,PSearch) else gtDistMode           # transform all ground truth distances with precomputed coefficients ["lin", "fit1", "fit2", "fit3"]
        self.corHistoryMode      = corHistoryMode.eval()      if isinstance(corHistoryMode      ,PSearch) else corHistoryMode       # method that is used to log epoch correlations to tensorboard ["spearman", "pearson"]

        self.lossFacMSE          = lossFacMSE.eval()          if isinstance(lossFacMSE          ,PSearch) else lossFacMSE           # loss weight for MSE term
        self.lossFacRelMSE       = lossFacRelMSE.eval()       if isinstance(lossFacRelMSE       ,PSearch) else lossFacRelMSE        # loss weight for relative MSE term
        self.lossFacPearsonCorr  = lossFacPearsonCorr.eval()  if isinstance(lossFacPearsonCorr  ,PSearch) else lossFacPearsonCorr   # loss weight for pearson correlation term
        self.lossFacSpearmanCorr = lossFacSpearmanCorr.eval() if isinstance(lossFacSpearmanCorr ,PSearch) else lossFacSpearmanCorr  # loss weight for spearman correlation term
        self.lossFacSlConvReg    = lossFacSlConvReg.eval()    if isinstance(lossFacSlConvReg    ,PSearch) else lossFacSlConvReg     # loss weight for regularizing the impact of slicing convolutions in the additive mode
        self.lossFacSizeReg      = lossFacSizeReg.eval()      if isinstance(lossFacSizeReg      ,PSearch) else lossFacSizeReg       # loss weight for size regularization
        self.lossSizeExp         = lossSizeExp.eval()         if isinstance(lossSizeExp         ,PSearch) else lossSizeExp          # exponent for size regularization
        self.lossOnlineMean      = lossOnlineMean.eval()      if isinstance(lossOnlineMean      ,PSearch) else lossOnlineMean       # if a sliced correlation computation for the loss uses a precomputed or online mean
        self.lossCorrAggregate   = lossCorrAggregate.eval()   if isinstance(lossCorrAggregate   ,PSearch) else lossCorrAggregate    # if a sliced correlation computation for the loss uses a running correlation computation or only the current value
        self.sampleSlicing       = sampleSlicing.eval()       if isinstance(sampleSlicing       ,PSearch) else sampleSlicing        # the 55 simulation pairs are sliced in subsets determined by this factor (should be in [1,5,11,55])
        self.calibEpochs         = calibEpochs.eval()         if isinstance(calibEpochs         ,PSearch) else calibEpochs          # number of epochs for mean norm accumulator coomputation
        self.calibIgnore         = calibIgnore.eval()         if isinstance(calibIgnore         ,PSearch) else calibIgnore          # if these strings are included in the path of a data sample, it is ignored for the mean norm accumulator coomputation

        self.dataAugmentation    = dataAugmentation.eval()    if isinstance(dataAugmentation    ,PSearch) else dataAugmentation     # if simple data augmentation (random rot, flip, channel swap) are used
        self.dataNormQuant       = dataNormQuant.eval()       if isinstance(dataNormQuant       ,PSearch) else dataNormQuant        # quantile to normalize the data to (e.g. 0.95 mean 95% of all data is in range [dataNormMin, dataNormMax])
        self.dataNormMin         = dataNormMin.eval()         if isinstance(dataNormMin         ,PSearch) else dataNormMin          # minimum for quantile data normalization
        self.dataNormMax         = dataNormMax.eval()         if isinstance(dataNormMax         ,PSearch) else dataNormMax          # maximum for quantile data normalization
        self.dataCrop            = dataCrop.eval()            if isinstance(dataCrop            ,PSearch) else dataCrop             # training data is randomly cropped to exactly this size
        self.dataCropRandom      = dataCropRandom.eval()      if isinstance(dataCropRandom      ,PSearch) else dataCropRandom       # if training data should be cropped randomly between dataCrop and original size
        self.dataScaleInference  = dataScaleInference.eval()  if isinstance(dataScaleInference  ,PSearch) else dataScaleInference   # data size that should be used for inference
        self.dataConvertMode     = dataConvertMode.eval()     if isinstance(dataConvertMode     ,PSearch) else dataConvertMode      # additional data conversions replace the data or are added to it
        self.dataCutoffIndex     = dataCutoffIndex.eval()     if isinstance(dataCutoffIndex     ,PSearch) else dataCutoffIndex      # only the first dataCutoffIndex simulation pairs from one data sample

        self.mBase               = mBase.eval()               if isinstance(mBase               ,PSearch) else mBase                # defines the architecture of the base network 
        self.mLinInit            = mLinInit.eval()            if isinstance(mLinInit            ,PSearch) else mLinInit             # value to initialize the weighted avg layers
        self.mLinDropout         = mLinDropout.eval()         if isinstance(mLinDropout         ,PSearch) else mLinDropout          # if dropout is used in the weighted avg layers
        self.mBaseInit           = mBaseInit.eval()           if isinstance(mBaseInit           ,PSearch) else mBaseInit            # method  to initialize the base network
        self.mFeatDist           = mFeatDist.eval()           if isinstance(mFeatDist           ,PSearch) else mFeatDist            # method to compute the latent space difference
        self.mNormMode           = mNormMode.eval()           if isinstance(mNormMode           ,PSearch) else mNormMode            # method to perform the feature map normalization
        self.mIgnoreLayers       = mIgnoreLayers.eval()       if isinstance(mIgnoreLayers       ,PSearch) else mIgnoreLayers        # layers of the base network that should be ignored for the distance computation

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.batch               = d.get("batch",                -1)
        p.epochs              = d.get("epochs",               -1)
        p.lrBase              = d.get("lrBase",               -1)
        p.lrAvg               = d.get("lrAvg",                -1)
        p.lrDecFac            = d.get("lrDecFac",             -1)
        p.lrDecTimes          = d.get("lrDecTimes",           -1)
        p.weightDecay         = d.get("weightDecay",          -1)
        p.gradClip            = d.get("gradClip",             -1)
        p.gtDistMode          = d.get("gtDistMode",           "")
        p.corHistoryMode      = d.get("corHistoryMode",       "")
        p.lossFacMSE          = d.get("lossFacMSE",           -1)
        p.lossFacRelMSE       = d.get("lossFacRelMSE",        -1)
        p.lossFacPearsonCorr  = d.get("lossFacPearsonCorr",   -1)
        p.lossFacSpearmanCorr = d.get("lossFacSpearmanCorr",  -1)
        p.lossFacSlConvReg    = d.get("lossFacSlConvReg",     -1)
        p.lossFacSizeReg      = d.get("lossFacSizeReg",       -1)
        p.lossSizeExp         = d.get("lossSizeExp",          -1)
        p.lossOnlineMean      = d.get("lossOnlineMean",       True)
        p.lossCorrAggregate   = d.get("lossCorrAggregate",    False)
        p.sampleSlicing       = d.get("sampleSlicing",        -1)
        p.calibEpochs         = d.get("calibEpochs",          -1)
        p.calibIgnore         = d.get("calibIgnore",          [])
        p.dataAugmentation    = d.get("dataAugmentation",     -1)
        p.dataNormQuant       = d.get("dataNormQuant",        -1)
        p.dataNormMin         = d.get("dataNormMin",          -1)
        p.dataNormMax         = d.get("dataNormMax",          -1)
        p.dataCrop            = d.get("dataCrop",             -1)
        p.dataCropRandom      = d.get("dataCropRandom",       True)
        p.dataScaleInference  = d.get("dataScaleInference",   -1)
        p.dataConvertMode     = d.get("dataConvertMode",      "")
        p.dataCutoffIndex     = d.get("dataCutoffIndex",      -1)
        p.mBase               = d.get("mBase",                "")
        p.mLinInit            = d.get("mLinInit",             -1)
        p.mLinDropout         = d.get("mLinDropout",          False)
        p.mBaseInit           = d.get("mBaseInit",            "")
        p.mFeatDist           = d.get("mFeatDist",            "")
        p.mNormMode           = d.get("mNormMode",            "")
        p.mIgnoreLayers       = d.get("mIgnoreLayers",        [])
        return p

    def asDict(self) -> dict:
        return {
            "batch"               : self.batch,
            "epochs"              : self.epochs,
            "lrBase"              : self.lrBase,
            "lrAvg"               : self.lrAvg,
            "lrDecFac"            : self.lrDecFac,
            "lrDecTimes"          : self.lrDecTimes,
            "weightDecay"         : self.weightDecay,
            "gradClip"            : self.gradClip,
            "gtDistMode"          : self.gtDistMode,
            "corHistoryMode"      : self.corHistoryMode,
            "lossFacMSE"          : self.lossFacMSE,
            "lossFacRelMSE"       : self.lossFacRelMSE,
            "lossFacPearsonCorr"  : self.lossFacPearsonCorr,
            "lossFacSpearmanCorr" : self.lossFacSpearmanCorr,
            "lossFacSlConvReg"    : self.lossFacSlConvReg,
            "lossFacSizeReg"      : self.lossFacSizeReg,
            "lossSizeExp"         : self.lossSizeExp,
            "lossCorrAggregate"   : self.lossCorrAggregate,
            "lossOnlineMean"      : self.lossOnlineMean,
            "sampleSlicing"       : self.sampleSlicing,
            "calibEpochs"         : self.calibEpochs,
            "calibIgnore"         : self.calibIgnore,
            "dataAugmentation"    : self.dataAugmentation,
            "dataNormQuant"       : self.dataNormQuant,
            "dataNormMin"         : self.dataNormMin,
            "dataNormMax"         : self.dataNormMax,
            "dataCrop"            : self.dataCrop,
            "dataCropRandom"      : self.dataCropRandom,
            "dataScaleInference"  : self.dataScaleInference,
            "dataConvertMode"     : self.dataConvertMode,
            "dataCutoffIndex"     : self.dataCutoffIndex,
            "mBase"               : self.mBase,
            "mLinInit"            : self.mLinInit,
            "mLinDropout"         : self.mLinDropout,
            "mBaseInit"           : self.mBaseInit,
            "mFeatDist"           : self.mFeatDist,
            "mNormMode"           : self.mNormMode,
            "mIgnoreLayers"       : self.mIgnoreLayers,
        }

# abstract base class for all parameter search classes below
class PSearch(object):
    def eval(self):
        raise NotImplementedError("Subclasses need to override this method!")

class PRange(PSearch):
    def __init__(self, pMin, pMax):
        self.pMin = pMin
        self.pMax = pMax
    def eval(self):
        if type(self.pMin) is int and type(self.pMax) is int:
            return random.randint(self.pMin, self.pMax)
        elif type(self.pMin) is float and type(self.pMax) is float:
            return random.uniform(self.pMin, self.pMax)
        else:
            raise ValueError("Invalid types!")

class PLogRange(PSearch):
    def __init__(self, start:float, stop:float, num:int, base:float=10.0):
        self.start = start
        self.stop = stop
        self.num = num
        self.base = base
    def eval(self):
        temp = np.logspace(self.start, self.stop, self.num, base=self.base)
        return random.choice(temp)


class PChoice(PSearch):
    def __init__(self, choices : list):
        self.choices = choices
    def eval(self):
        return random.choice(self.choices)
