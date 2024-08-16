# Bubble Localization and Rendering for the SBC

![Flow chart of steps in bubble localization for the SBC chambers.](thesis/figures/localization_flow_chart_2.png)

This is the repository for the code associated with my thesis that can be found [here](thesis/Bubble_Localization_and_Rendering_for_the_SBC.pdf).

The primary files are [setup.py](setup.py) which contains functions for setting up and rendering images of the chambers, and [utils.py](utils.py) which contains several functions that are used for triangulation, remapping, creating plots, etc. Examples of how the functions from those files are used to create renders and triangulate bubbles can be found within the Jupyter notebooks.

The associated Blender files used to clean up and export the chamber model can be found [here](https://drive.google.com/drive/folders/1IG_XLitbM6101vEFMDmiwda3bSZK6OpN?usp=drive_link). If you wish to make edits to any of the chamber elements, you can do so in Blender and export the selected components to the `chamber_model/components/` folder. Make sure to make any corresponding necessary changes to the `setup.py` file!

## Setting Up the Environment

The code in this repository was run in an Anaconda environment on a windows machine with a CUDA-enabled NVIDIA GPU. If you do not already have a version of Anaconda installed, I recommend downloading a minimal version of it [here](https://docs.anaconda.com/miniconda/) (or download a version with a full GUI from [here](https://www.anaconda.com/download/success)). To set up a conda environment on a windows machine using the provided `environment.yml` file, see [Windows](#windows). If that does not work, or if you are using a non-windows OS, see [Manually Installing Packages](#manually-installing-packages).

### Windows

To setup the conda environment using the `environment.yml` file, navigate to this repository's directory in an Anaconda Prompt and run the following:

```console
conda env create -f environment.yml
```

This will create a new conda environment named `Bubble` and automatically install all required packages with the versions used to create this code.

### Manually Installing Packages

You can also install all the required packages using the provided `requirements.txt` file. If using a python virtual env, you should be able to install all required packages using just:

```console
pip install -r requirements.txt
```

If using a conda environment, first create a new pip installation in the environment then install the required packages with pip:

```console
conda install pip
pip install -r requirements.txt
```

Note that we run `conda install pip` as it makes sure the packages are installed for the conda environment and not to the system-wide environment.

### Running Notebooks

If you are using a code editor like VSCode, you should be able to open the Jupyter notebooks and run them using the newly created `Bubble` environment. If instead you want to run the code in the native Jupyter notebook/lab environment, first activate the environment in an Anaconda Prompt:

```console
conda activate Bubble
```

Then activate `Jupyter Lab` using the command

```console
jupyter lab
```

which should open a lab environment where you can edit and run the notebooks using your default browswer.

## Notebooks

- [`BasicRender.ipynb`](BasicRender.ipynb) Walks through the basics of setting up a Mitsuba scene of the chamber and creating a render.
- [`PixelRemappingAndTriangulation.ipynb`](PixelRemappingAndTriangulation.ipynb) Creates renders of the testing grid and uses those renders to determine the remapping functions and triangulations. This was the primary notebook used to test different remapping functions, triangulation methods, and optimization techniques.
- [`PixelRemappingAndTriangulation_Wrong_IoRs.ipynb`](PixelRemappingAndTriangulation_Wrong_IoRs.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that creates renders of a smaller test grid to determine the effect that errors in the indices of refraction of chamber elements have on triangulation errors.
- [`PixelRemappingAndTriangulation_Multiple.ipynb`](PixelRemappingAndTriangulation_Multiple.ipynb) A branch of the `PixelRemappingAndTriangulation.ipynb` notebook that tests all of the available optimization methods simultaneously and plots their resulting loss curves and triangulation error histograms.
- [`PnP_Test_Chamber.ipynb`](PnP_Test_Chamber.ipynb) Determines the pose of camera 2 of the chamber using the pixel positions of the fiducial markers in a real image.
- [`PoseMatrixEstimation.ipynb`](PoseMatrixEstimation.ipynb) Estimates the pose of the cameras used for rendering by using estimated pixel positions of the fiducial markers within each camera's image. It also saves the resulting estimated pose matrices.
- [`TriangulationWithPixelLocalizationError.ipynb`](TriangulationWithPixelLocalizationError.ipynb) Plots the relationship between the average error in the pixel localization of bubbles vs the resulting error in triangulation.
- [`TriangulationWithPixelOffsetPnPPoses.ipynb`](TriangulationWithPixelOffsetPnPPoses.ipynb) Plots the relationship between the error in pixel localization of the fiducial markers vs the resulting error in the estimated pose and average triangulation error.
- [`TriangulationWithPnPEstimatedPoseMatrix.ipynb`](TriangulationWithPnPEstimatedPoseMatrix.ipynb) Runs n-view triangulation on the test grid using the estimated poses calculated in `PoseMatrixEstimation.ipynb`.
- [`CorrespondingBubbleDetectionTest.ipynb`](CorrespondingBubbleDetectionTest.ipynb) Features a test for a possible solution to determining corresponding bubble locations between images.
- [`BulkRendering.ipynb](BulkRendering.ipynb) Shows how multiple renders of a grid of bubbles can be created using the `bubble_grid_renders` function.

---

Note: notebooks located in the archive folder are not well documented and feature code that may be deprecated. They were mainly used in initial testing.

- [`archive\PnPEstimationWithCheckerboard.ipynb`](archive\PnPEstimationWithCheckerboard.ipynb) Tries to estimate the pose matrices of the cameras in the chamber by placing a checkerboard in the scene where the fiducial markers would normally be.
- [`archive\PnP_Test_Render.ipynb`](archive\PnP_Test_Render.ipynb) An earlier version of `archive\PnPEstimationWithCheckerboard.ipynb` that computed the pose for a single camera by placing the checkerboard in the scene where the fiducial markers would be.
- [`archive\CameraCalibrationWithRenderedCheckerboard.ipynb`](archive\CameraCalibrationWithRenderedCheckerboard.ipynb) Calibrated the camera used to render images by rendering a series of images of a checkerboard. Also features code that could be adapted to calibrate real cameras with real images.
- [`archive\LocationRemappingTest.ipynb`](archive\LocationRemappingTest.ipynb) Features some of the initial tests for determining the remapping coefficients of distortion functions.
- [`archive\LocationMappingComparedWithAndWithoutDistortedSurface.ipynb`](archive\LocationMappingComparedWithAndWithoutDistortedSurface.ipynb) Similar to `archive\LocationRemappingTest.ipynb`, performs initial tests for pixel remapping, now comparing the results with and without the distorted jar surfaces.
- [`archive\TwoCameraTriangulationError.ipynb`](archive\TwoCameraTriangulationError.ipynb) Initial triangulation tests for two-camera midpoint triangulation.
- [`archive\PoseOptimizationWithRenders.ipynb`](archive\PoseOptimizationWithRenders.ipynb) Attempts to use Mitsuba's inverse rendering to optimize for the pose of the cameras. Based on [this Mitsuba tutorial](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/object_pose_estimation.html).
- [`archive\BasicChamberSurfaceOptimizationRender.ipynb`](archive\BasicChamberSurfaceOptimizationRender.ipynb) Attempts at optimizing a heightmap to determine jar surface distortion. Optimizations were attempted by comparing to a rendered image with a distorted jar surface. Based on [this Mitsuba tutorial](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/caustics_optimization.html).