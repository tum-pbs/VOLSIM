

class LrDecayPolicy(object):
    def __init__(self, lrDecayFactor, lrDecayTimes, maxEpoch):
        self.lrDecayFactor = lrDecayFactor
        self.lrDecayTimes = lrDecayTimes
        self.maxEpoch = maxEpoch
    def __call__(self, epoch):
        return self.lrDecayFactor ** (epoch * (self.lrDecayTimes/(self.maxEpoch + 0.0000001)) )