import torch
import torch.nn as nn
from typing import Tuple



class MultiScaleNet(torch.nn.Module):
    def __init__(self, widthFactor=1, layers=12, firstChannels=3, useSkip=False):
        super(MultiScaleNet, self).__init__()
        assert layers in [12,16,20]
        self.widthFactor = widthFactor
        self.channels = [int(4 * widthFactor),
                        int(4 * widthFactor),
                        int(4 * widthFactor),
                        int(4 * widthFactor),
                        int(8 * widthFactor),
                        int(8 * widthFactor),
                        int(8 * widthFactor),
                        int(8 * widthFactor),
                        int(16 * widthFactor),
                        int(16 * widthFactor),
                        int(16 * widthFactor),
                        int(16 * widthFactor),]
        if layers > 12:
            self.channels += [int(32 * widthFactor),
                            int(32 * widthFactor),
                            int(32 * widthFactor),
                            int(32 * widthFactor),]
        if layers > 16:
            self.channels += [int(64 * widthFactor),
                            int(64 * widthFactor),
                            int(64 * widthFactor),
                            int(64 * widthFactor),]
        self.layers = layers
        self.layerList = []
        self.useSkip = useSkip


        # native res
        self.slice0 = nn.Sequential(
            nn.Conv3d(firstChannels, self.channels[0], 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice0]

        self.slice1 = torch.nn.Sequential(
            nn.Conv3d(self.channels[0], self.channels[1], 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice1]

        self.slice2 = torch.nn.Sequential(
            nn.Conv3d(self.channels[1], self.channels[2], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice2]

        self.slice3 = torch.nn.Sequential(
            nn.Conv3d(self.channels[2], self.channels[3], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice3]


        # 1/2 res
        self.pool0 = nn.AvgPool3d(2, stride=2, padding=0)
        blockChannels = firstChannels if not self.useSkip else firstChannels + self.channels[3]
        self.slice4 = torch.nn.Sequential(
            nn.Conv3d(blockChannels, self.channels[4], 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice4]

        self.slice5 = torch.nn.Sequential(
            nn.Conv3d(self.channels[4], self.channels[5], 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice5]

        self.slice6 = torch.nn.Sequential(
            nn.Conv3d(self.channels[5], self.channels[6], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice6]

        self.slice7 = torch.nn.Sequential(
            nn.Conv3d(self.channels[6], self.channels[7], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice7]


        # 1/4 res
        self.pool1 = nn.AvgPool3d(4, stride=4, padding=0)
        blockChannels = firstChannels if not self.useSkip else firstChannels + self.channels[7]
        self.slice8 = torch.nn.Sequential(
            nn.Conv3d(blockChannels, self.channels[8], 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice8]

        self.slice9 = torch.nn.Sequential(
            nn.Conv3d(self.channels[8], self.channels[9], 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice9]

        self.slice10 = torch.nn.Sequential(
            nn.Conv3d(self.channels[9], self.channels[10], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice10]

        self.slice11 = torch.nn.Sequential(
            nn.Conv3d(self.channels[10], self.channels[11], 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layerList += [self.slice11]


        # 1/8 res
        if self.layers > 12:
            self.pool2 = nn.AvgPool3d(8, stride=8, padding=0)
            blockChannels = firstChannels if not self.useSkip else firstChannels + self.channels[11]
            self.slice12 = torch.nn.Sequential(
                nn.Conv3d(blockChannels, self.channels[12], 5, stride=1, padding=2),
                nn.ReLU(),
            )
            self.layerList += [self.slice12]

            self.slice13 = torch.nn.Sequential(
                nn.Conv3d(self.channels[12], self.channels[13], 5, stride=2, padding=2),
                nn.ReLU(),
            )
            self.layerList += [self.slice13]

            self.slice14 = torch.nn.Sequential(
                nn.Conv3d(self.channels[13], self.channels[14], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice14]

            self.slice15 = torch.nn.Sequential(
                nn.Conv3d(self.channels[14], self.channels[15], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice15]


        # 1/16 res
        if self.layers > 16:
            self.pool3 = nn.AvgPool3d(16, stride=16, padding=0)
            blockChannels = firstChannels if not self.useSkip else firstChannels + self.channels[15]
            self.slice16 = torch.nn.Sequential(
                nn.Conv3d(blockChannels, self.channels[16], 5, stride=1, padding=2),
                nn.ReLU(),
            )
            self.layerList += [self.slice16]

            self.slice17 = torch.nn.Sequential(
                nn.Conv3d(self.channels[16], self.channels[17], 5, stride=2, padding=2),
                nn.ReLU(),
            )
            self.layerList += [self.slice17]

            self.slice18 = torch.nn.Sequential(
                nn.Conv3d(self.channels[17], self.channels[18], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice18]

            self.slice19 = torch.nn.Sequential(
                nn.Conv3d(self.channels[18], self.channels[19], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice19]



    def forward(self, X:torch.Tensor) -> Tuple[torch.Tensor, list]:
        out = []
        h_in = X

        # native res
        h = self.slice0(X)
        h_relu1 = h
        out += [h_relu1]

        h = self.slice1(h)
        h_relu2 = h
        out += [h_relu2]

        h = self.slice2(h)
        h_relu3 = h
        out += [h_relu3]

        h = self.slice3(h)
        h_relu4 = h
        out += [h_relu4]


        # 1/2 res
        pooled0 = self.pool0(X)
        if self.useSkip:
            pooled0 = torch.cat((pooled0, h_relu4), 1)
        h = self.slice4(pooled0)
        h_relu5 = h
        out += [h_relu5]

        h = self.slice5(h)
        h_relu6 = h
        out += [h_relu6]

        h = self.slice6(h)
        h_relu7 = h
        out += [h_relu7]

        h = self.slice7(h)
        h_relu8 = h
        out += [h_relu8]


        # 1/4 res
        pooled1 = self.pool1(X)
        if self.useSkip:
            pooled1 = torch.cat((pooled1, h_relu8), 1)
        h = self.slice8(pooled1)
        h_relu9 = h
        out += [h_relu9]

        h = self.slice9(h)
        h_relu10 = h
        out += [h_relu10]

        h = self.slice10(h)
        h_relu11 = h
        out += [h_relu11]

        h = self.slice11(h)
        h_relu12 = h
        out += [h_relu12]


        # 1/8 res
        if self.layers > 12:
            pooled2 = self.pool2(X)
            if self.useSkip:
                pooled2 = torch.cat((pooled2, h_relu12), 1)
            h = self.slice12(pooled2)
            h_relu13 = h
            out += [h_relu13]

            h = self.slice13(h)
            h_relu14 = h
            out += [h_relu14]

            h = self.slice14(h)
            h_relu15 = h
            out += [h_relu15]

            h = self.slice15(h)
            h_relu16 = h
            out += [h_relu16]


        # 1/16 res
        if self.layers > 16:
            pooled3 = self.pool3(X)
            if self.useSkip:
                pooled3 = torch.cat((pooled3, h_relu16), 1) #type:ignore
            h = self.slice16(pooled3)
            h_relu17 = h
            out += [h_relu17]

            h = self.slice17(h)
            h_relu18 = h
            out += [h_relu18]

            h = self.slice18(h)
            h_relu19 = h
            out += [h_relu19]

            h = self.slice19(h)
            h_relu20 = h
            out += [h_relu20]

        #print(h_in.shape)
        #print(h_relu1.shape)
        #print(h_relu2.shape)
        #print(h_relu3.shape)
        #print(h_relu4.shape)
        #print("  " + str(h_relu5.shape))
        #print("")

        return out




class AlexNetLike(torch.nn.Module):
    def __init__(self, widthFactor=1, layers=5, convKernel=11, maxPoolKernel=3, firstStride=4, firstChannels=3, useRacecar=False):
        super(AlexNetLike, self).__init__()
        self.useRacecar = useRacecar
        self.widthFactor = widthFactor
        self.channels = [int(32 * widthFactor),
                         int(96 * widthFactor),
                        int(192 * widthFactor),
                        int(128 * widthFactor),
                        int(128 * widthFactor)]
        self.layers = layers

        if convKernel == 11 and maxPoolKernel == 3:
            self.featureMapSize = [31,15,7,7,7]
            rcOutputPad = 1
        elif convKernel == 12 and maxPoolKernel == 4:
            self.featureMapSize = [31,14,6,6,6]
            rcOutputPad = 0
        elif convKernel == 9 and maxPoolKernel == 2:
            self.featureMapSize = [10,5,2,2,2]
            rcOutputPad = 0
        else:
            raise ValueError("Unknown base network configuration")

        self.layerList = []

        self.slice0 = nn.Sequential(
            nn.Conv3d(firstChannels, self.channels[0], convKernel, stride=firstStride, padding=2),
            nn.ReLU(),
        )
        self.layerList += [self.slice0]

        if layers > 1:
            self.slice1 = torch.nn.Sequential(
                nn.MaxPool3d(maxPoolKernel, stride=2, padding=0),
                nn.Conv3d(self.channels[0], self.channels[1], 5, stride=1, padding=2),
                nn.ReLU(),
            )
            self.layerList += [self.slice1]

        if layers > 2:
            self.slice2 = torch.nn.Sequential(
                nn.MaxPool3d(maxPoolKernel, stride=2, padding=0),
                nn.Conv3d(self.channels[1], self.channels[2], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice2]

        if layers > 3:
            self.slice3 = torch.nn.Sequential(
                nn.Conv3d(self.channels[2], self.channels[3], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice3]

        if layers > 4:
            self.slice4 = torch.nn.Sequential(
                nn.Conv3d(self.channels[3], self.channels[4], 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.layerList += [self.slice4]


    def forward(self, X:torch.Tensor) -> Tuple[torch.Tensor, list]:
        out = []
        h_in = X

        h = self.slice0(X)
        h_relu1 = h
        out += [h_relu1]

        if self.layers > 1:
            h = self.slice1(h)
            h_relu2 = h
            out += [h_relu2]

        if self.layers > 2:
            h = self.slice2(h)
            h_relu3 = h
            out += [h_relu3]

        if self.layers > 3:
            h = self.slice3(h)
            h_relu4 = h
            out += [h_relu4]

        if self.layers > 4:
            h = self.slice4(h)
            h_relu5 = h
            out += [h_relu5]

        #print(h_in.shape)
        #print(h_relu1.shape)
        #print(h_relu2.shape)
        #print(h_relu3.shape)
        #print(h_relu4.shape)
        #print("  " + str(h_relu5.shape))
        #print("")

        return out