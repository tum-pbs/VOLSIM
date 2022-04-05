## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from volsim.distance_model import *


## SETUP
useGPU = True

modelVolsim = DistanceModel.load("models/VolSiM.pth", useGPU=useGPU)
# freeze volsim weights
for param in modelVolsim.parameters():
    param.requires_grad = False
print()

adv = np.load("data/64_example/advdiff/sim_000000/density_000120.npz")['arr_0'].repeat(3, axis=4)
stacked = np.stack([adv[0], adv[5], adv[10]]) # take some elements from the sequence
stacked = torch.from_numpy(stacked).permute((0,4,1,2,3)).float()
stacked = (stacked - stacked.mean()) / stacked.std()
dataset = TensorDataset(stacked.cuda()) if useGPU else TensorDataset(stacked.cpu())
loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True)


## AUTOENCODER THAT RECONSTRUCTS FROM A COMPRESSION WITH FACTOR 4
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        eW = 64
        self.encConv = nn.Sequential(
            nn.Conv3d(3, eW, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(eW, eW, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(eW, 2*eW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*eW, 2*eW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*eW, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        dW = 64
        self.decConv = nn.Sequential(
            nn.Conv3d(3, dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2,2), mode='nearest'),
            nn.Conv3d(dW, dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2,2), mode='nearest'),
            nn.Conv3d(dW, 2*dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*dW, 2*dW, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*dW, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, data:torch.Tensor) -> torch.Tensor:
        latent = self.encConv(data)
        result = self.decConv(latent)
        return result


## VOLSIM LOSS FUNCTION
# input shape: [batch, channel, width, height, depth]  -> output shape: [batch, channel]
def loss_volsim(x:torch.Tensor, y:torch.Tensor, combinedNorm:bool) -> torch.Tensor:

    # stack along new dimension as volsim select data from a single tensor based on indices
    data = torch.stack([x,y], dim=1)
    # move channel dimension to back
    data = data.permute(0,1,3,4,5,2)

    # normalize each intput channel to volsim input range [-1,1]
    if combinedNorm: # combined normalization of both inputs
        dMin = torch.amin(data, dim=(1,2,3,4), keepdim=True)
        dMax = torch.amax(data, dim=(1,2,3,4), keepdim=True)
    else: # separate normalization of both inputs
        dMin = torch.amin(data, dim=(2,3,4), keepdim=True)
        dMax = torch.amax(data, dim=(2,3,4), keepdim=True)
    data = 2 * ((data - dMin) / (dMax - dMin)) - 1

    # create indices to compare the [0, nPairs) elements from x with the [nPairs, 2*nPairs) from y
    nPairs = 1
    indexA = torch.unsqueeze(torch.arange(0, nPairs), dim=0)
    indexB = torch.unsqueeze(torch.arange(nPairs, 2*nPairs), dim=0)

    # create dict and compute distance
    inDict = {"data": data, "indexA": indexA, "indexB": indexB, "idxMin": 0, "idxMax": indexA.shape[1]}
    distance = modelVolsim(inDict)

    return distance



## OPTIMIZATION
model = SimpleAE().cuda() if useGPU else SimpleAE().cpu()
parameters = filter(lambda p: p.requires_grad, model.parameters())
print("Parameter: %d" % sum([np.prod(p.size()) for p in parameters]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for i in range(100):
    for j, sample in enumerate(loader, 0):

        optimizer.zero_grad()
        gt = sample[0]
        rec = model(gt)
        loss = F.mse_loss(rec, gt) + 0.3*loss_volsim(rec, gt, combinedNorm=True).mean()

        loss.backward()
        optimizer.step()

        print("[%2d, %2d]  %1.5f" % (i,j,loss.item()))


## VISUALIZE RECONSTRUCTIONS
with torch.no_grad():
    model.eval()
    names = ["Adv0", "Adv5", "Adv10"]
    for i, sample in enumerate(dataset, 0):
        gt = sample[0].unsqueeze(0)
        rec = model(gt)

        gt = gt.permute((2,3,4,1,0)).squeeze().cpu().numpy()
        gt = np.mean(gt, axis=2)
        gtMin = np.min(gt)
        gtMax = np.max(gt)
        gt = (gt - gtMin) / (gtMax - gtMin)

        rec = rec.permute((2,3,4,1,0)).squeeze().cpu().numpy()
        rec = np.mean(rec, axis=2)
        recMin = np.min(rec)
        recMax = np.max(rec)
        rec = (rec - recMin) / (recMax - recMin)

        fig, ax = plt.subplots(1,2, figsize=(5,2.5), dpi=200)
        fig.add_subplot(111, frameon=False)
        fig.subplots_adjust(wspace=0.03, hspace=0.06)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

        ax[0].imshow(gt)
        ax[0].set_title("%s - Ground Truth" % names[i])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(rec)
        ax[1].set_title("%s - Reconstruction" % names[i])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.savefig("results/Rec_%s.png" % names[i], bbox_inches='tight')

print("Reconstruction visualizations written to Results/")
