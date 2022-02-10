# Volumetric Similarity Metric (VolSiM) for Vectorial and Scalar 3D Data
This repository contains the source code for the paper [Learning Similarity Metrics for Volumetric Simulations with Multiscale CNNs](https://arxiv.org/abs/2202.04109) by [Georg Kohl](https://ge.in.tum.de/about/georg-kohl/), [Liwei Chen](https://ge.in.tum.de/about/dr-liwei-chen/), and [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/).

*VolSiM* is a metric intended as a comparison method for dense, volumetric, vectorial or scalar data from numerical simulations. It computes a scalar distance value from two inputs that indicates the similarity between them, where a higher value indicates stronger differences. Traditional metrics like L<sup>1</sup> or L<sup>2</sup> distances or the peak signal-to-noise ratio (PSNR) are suboptimal comparison methods for simulation data, as they only consider element-wise comparisons and cannot capture structures on different scales or contextual information. For example, consider a volumetric checkerboard pattern and a version that is translated by one voxel along one dimension. Comparing both element-wise leads to a large distance as all voxels are very different, even though the structure of both patterns is identical. Instead of comparing element-wise, *VolSiM* extracts deep feature maps with a multiscale CNN structure from both inputs and compares them. This means similarity on different scales and recurring structures or patterns are considered in the distance evaluation.

Further information is available at our [project website](https://ge.in.tum.de/publications/2022-volsim-kohl/). To compare scalar 2D data, you can have a look at our CNN-based metric [*LSiM*](https://github.com/tum-pbs/LSIM) that was specifically designed for this data domain.

-----------------------------------------------------------------------------------------------------

## Installation
In the following, Linux is assumed as the OS but the installation on Windows should be similar. First, clone this repository to a destination of your choice.
```
git clone https://github.com/tum-pbs/VOLSIM
cd VOLSIM
```
We recommend to install the required python packages (see `requirements.txt`) via a conda environment (e.g. using [miniconda](https://docs.conda.io/en/latest/miniconda.html)), but it may be possible to install them with *pip* (e.g. via *venv* for a separate environment) as well.
```
conda create --name VOLSIM_Env --file requirements.txt --channel default --channel pytorch
conda activate VOLSIM_Env
```
To test if the installation was successful, run `distance_example_simple.py` (see below) and check if the distance output matches with the comment at the bottom of the script. In the following all commands should be run from the root folder of the repository. If you encounter problems with installing, training, or evaluating the metric, let us know by opening an [issue](https://github.com/tum-pbs/VOLSIM/issues).

## Basic Usage
To evaluate the metric on two numpy arrays `arr1, arr2` you only need to load the model and call the `computeDistance` method. Supported input shapes are `[width, height, depth, channels]` or `[batch, width, height, depth, channels]`, with one or three channels.
```python
from volsim.distance_model import *
model = DistanceModel.load("models/VolSiM.pth", useGPU=True)
dist = model.computeDistance(arr1, arr2, normalize=True, interpolate=False)
# resulting shapes: input -> output
# [width, height, depth, channel] -> [1]
# [batch, width, height, depth, channel] -> [batch]
```
The input processing can be modified via the parameters `normalize` and `interpolate`. The `normalize` argument indicates that both input arrays will be normalized to `[-1,1]` via a min-max normalization. In general, this setting is recommended as the metric CNN was trained on this value range, but if the data is already normalized before, it can be omitted. The `interpolate` argument determines if both inputs are interpolated to the input size of `64x64x64` on which the network was trained via a cubic spline interpolation. Since the model is fully convolutional, different input shapes are possible as well, and we determined that the metric still remains stable for spatial input dimensions between `32x32x32 - 128x128x128`. Outside this range the model performance may drop, and too small inputs can cause issues as the feature extractor spatially reduces the input dimensions.

The resulting numpy array `dist` contains distance values with shape `[1]` or `[batch]` depending on the shape of the inputs. If the evaluation should only use the CPU, set `useGPU=False` when loading the model. A simple example is shown in `distance_example_simple.py`, and `distance_example_detailed.py` shows a more advanced usage with a correlation evaluation. To run these examples use:
```
python src/distance_example_simple.py
python src/distance_example_detailed.py
```


-----------------------------------------------------------------------------------------------------

## Data Download and Processing
**[Archiving our data sets is still in progress, but a download link will be coming soon]**

`copy_data_lowres.py` can used to downsample the simulation resolution of `128x128x128` to the training and evaluation resolution of `64x64x64`. It processes all .npz data files, while creating copies of all supplementary files in the input directory.

To process custom raw simulation data, `compute_nonlinear_dist_coef.py` can be used to compute the nonlinear distance coefficients that are required for the ground truth distances from the proposed entropy-based similarity model. It creates a .json file with a path to each data file and a corresponding distance coefficient value. `plot_data_vis.py` contains simple plotting functionality to visualize individual data samples and the corresponding ground truth distances.


## Metric Comparison
Once the data sets are available to download, the performance of different metrics (element-wise and CNN-based) on our data can be compared using the metric evaluations in `eval_metrics_shallow_tb.py` and `eval_metrics_trained_tb.py`:
```
python src/eval_metrics_shallow_tb.py
python src/eval_metrics_trained_tb.py
```
Both scripts compute distances for various metrics on our data sets, and evaluate them against the proposed ground truth distance model. All results are printed in the console and in addition written to tensorboard event files in the `runs` directory. The event files can be read by opening tensorboard via `tensorboard --logdir=runs`. Running the metric evaluation without changes should result in values similar to Table 1 in our paper.

## Re-training the Model
Once the data sets are available to download, the metric can be re-trained from scratch via `training.py`:
```
python src/training.py
```
The training progress is printed in the console and in addition written to tensorboard event files in the `runs` directory. The event files can be read by opening tensorboard via `tensorboard --logdir=runs`. Running the training script without changes should result in a model with a performance close to our final *VolSiM* metric (when evaluated with the metric evaluation above). But of course, minor deviations are expected due to the random nature of the model initialization and training procedure. The training setup for different model variants are included in `training_iso.py` and as a commented set of parameters in `training.py`.

## Backpropagation through the Metric
Backpropagation through the metric network is straightforward by integrating the `DistanceModel` class that derives from `torch.nn.Module` in the target network. Load the trained model weights from the model directory with the `load` method in `DistanceModel` on initialization (see the basic usage above), and freeze all trainable weights of the metric if required. In this case, the `forward` method of the metric (via calling the model object and with appropriate data handling beforehand) must be used instead of `computeDistance` to perform the comparison operation.

-----------------------------------------------------------------------------------------------------

## Acknowledgements
This repository also contains the image-based LPIPS metric from the [perceptual similarity](https://github.com/richzhang/PerceptualSimilarity) repository for comparison.
