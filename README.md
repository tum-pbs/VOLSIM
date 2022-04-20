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

## Data Generation, Download, and Processing

### Downloading our data
**[Archiving our data sets is still in progress, but a download link will be coming soon]**

### Generation from MantaFlow
<details>
<summary>Click to expand detailed MantaFlow instructions</summary>

To generate data with the fluid solver [MantaFlow](http://mantaflow.com/), perform the following steps:
1. Download the [MantaFlow source code](https://github.com/tum-pbs/mantaflow) and follow the [installation instructions](http://mantaflow.com/install.html). **Our additional code assumes the usage of commit [3a74f09](https://github.com/tum-pbs/mantaflow/tree/3a74f0951ade7e7bb61515acd0cfdf9964757a73)! Newer commits might still work, but may cause problems.**
2. Ensure that numpy and imageio are installed in the python environment used for MantaFlow.
3. Add our implementation of some additional functionality to the solver by replacing the following files in your MantaFlow directory, then re-build the solver:
    - Replace `source/plugin/numpyconvert.cpp` with `data/generation_scripts/MantaFlow/source/numpyconvert.cpp` (for the copyArrayToGridInt and copyGridToArrayInt functions)
    - Replace `source/conjugategrad.cpp` with `data/generation_scripts/MantaFlow/source/conjugategrad.cpp` (for the ApplyMatrix1D and cgSolveDiffusion1D functions)
    - Replace  `source/test.cpp` with `data/generation_scripts/MantaFlow/source/test.cpp` (for the Advection-Diffusion and Burger's equation implementation, as well as various utilities)
4. Copy the `data/generation_scripts/MantaFlow/scripts3D` folder to the root of your MantaFlow directory.
5. This scripts folder contains the MantaFlow scene files for each data set (.py files), that can be run in the same way as normal MantaFlow scene files. The corresponding batch generation scripts (.sh files) simply run each scene multiple times with different parameters to build a full data set. If one batch file creates different data sets, e.g. a training and a test set variant, you can find each set of parameters as a comment in the batch file.
6. As the liquid and smoke generation has to run an individual simulation for each sequence element, the `data/generation_scripts/MantaFlow/scripts3D/compactifyData.py` scene file combines the existing individual numpy arrays to ensure a coherent data set structure. It should be run like other scene files as a post-processing step once the liquid or smoke generation is complete.
</details>



### Generation from PhiFlow
<details>
<summary>Click to expand detailed PhiFlow instructions</summary>

To generate data with the fluid solver [PhiFlow](https://ge.in.tum.de/research/phiflow/), perform the following steps:
1. Download the [PhiFlow source code](https://github.com/tum-pbs/PhiFlow) and follow the [installation instructions](https://tum-pbs.github.io/PhiFlow/Installation_Instructions.html), using the custom CUDA kernels is highly recommended for performance reasons. **Our additional code assumes the usage of commit [f3090a6](https://github.com/tum-pbs/PhiFlow/tree/f3090a6963a2dc08df9fb39ce270cc30107d69b6)! Substantially newer commits will not work, due to larger architecture changes in following versions.**
2. Ensure that numpy and imageio are installed in the python environment used for PhiFlow.
3. Add our implementation of some additional functionality to the solver by copying the all files from the `data/generation_scripts/PhiFlow` folder to the `demos` folder in your PhiFlow directory.
4. The copied files contain the PhiFlow scene files for each data set (.py files), that can be run in the same way as normal PhiFlow scene files in the `demos` folder. Note, that the `data/generation_scripts/PhiFlow/varied_sim_utils.py` file only contains import utilities and can not be run individually. The corresponding batch generation scripts (.sh files) simply run each scene multiple times with different parameters to build a full data set. If one batch file creates different data sets, e.g. a training and a test set variant, you can find each set of parameters as a comment in the batch file.
</details>



### Generation from the Johns Hopkins Turbulence Database (JHTDB)
<details>
<summary>Click to expand detailed JHTDB instructions</summary>

To extract sequences from the [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu/), the required steps are:
1. Install the [pyJHTDB package](https://github.com/idies/pyJHTDB) for local usage.
2. Request an [authorization token](http://turbulence.pha.jhu.edu/authtoken.aspx) to ensure access to the full data base.
3. Add your authorization token to the script `data/generation_scripts/convert_JHTDB.py`, adjust the settings as necessary, and run the script to download and convert the corresponding regions of the DNS data.
</details>



### Generation from ScalarFlow
<details>
<summary>Click to expand detailed ScalarFlow instructions</summary>

To process the [ScalarFlow](https://ge.in.tum.de/publications/2019-scalarflow-eckert/) data set into sequences suitable for metric evaluations, the following steps are necessary:
1. Download the full data set from the [mediatum repository](https://mediatum.ub.tum.de/1521788) and extract it at the target destination.
2. Add the root folder of the extracted data set as the input path in the `data/generation_scripts/convert_scalarFlow.py` script.
3. Adjust the conversion settings like output path or resolution in the script if necessary, and run it to generate the data set.
</details>



### General Data Post-Processing
`plot_data_vis.py` contains simple plotting functionality to visualize individual data samples and the corresponding ground truth distances. `copy_data_lowres.py` can be used to downsample the generation resolution of `128x128x128` to the training and evaluation resolution of `64x64x64`. It processes all .npz data files, while creating copies of all supplementary files in the input directory.

To process custom raw simulation data, `compute_nonlinear_dist_coef.py` can be used to compute the nonlinear distance coefficients that are required for the ground truth distances from the proposed entropy-based similarity model. It creates a .json file with a path to each data file and a corresponding distance coefficient value.



-----------------------------------------------------------------------------------------------------

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
Backpropagation through the metric network is straightforward by integrating the `DistanceModel` class that derives from `torch.nn.Module` in the target network. Load the trained model weights from the model directory with the `load` method in `DistanceModel` on initialization (see Basic Usage above), and freeze all trainable weights of the metric if required. In this case, the metric model should be called directly (with appropriate data handling beforehand) instead of using `computeDistance` to perform the comparison operation. An example for this process based on a simple Autoencoder can be found in `backprop_example.py`:
```
python src/backprop_example.py
```


-----------------------------------------------------------------------------------------------------

## Acknowledgements
This repository also contains the image-based LPIPS metric from the [perceptual similarity](https://github.com/richzhang/PerceptualSimilarity) repository for comparison.
