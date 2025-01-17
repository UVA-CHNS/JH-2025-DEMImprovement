# JH-2025-DEMImprovement

# Deep Learning-Based Downscaling of Global Digital Elevation Models for Enhanced Urban Flood Modeling 

# Overview
This GitHub repository contains the source code and datasets used in the research paper, "Improving Spatial Resolution and 
This GitHub repository contains the source code and datasets used in the research paper, "Improving Spatial Resolution and Accuracy of Global Digital Elevation Models for Flood Modeling Using Deep Learning-Based Integration of Remote Sensing Data." The paper introduces a deep learning-based approach to generate high-resolution Digital Elevation Models (DEMs) by leveraging multi-source data, including GDEMs (ALOS, ASTER, NASADEM, SRTM), remote imagery data (Landsat-8, Sentinel-1, Sentinel-2), and OpenStreetMap layers. This method utilizes a Convolutional Neural Network (CNN) U-Net model to integrate and refine these diverse data sources, producing enhanced DEMs at a 5-meter resolution. This approach significantly improves the accuracy and resolution of global DEMs, leading to more precise models essential for urban flood management and simulations.

# Key Contributions
- A high-resolution DEM was developed for urban flood modeling using deep learning-based multi-source data integration.
- The enhanced DEM significantly improved accuracy, showing reductions in RMSE and MAE, and detailed urban feature representation.
- The improved DEM demonstrated better flood pattern accuracy by increasing detection probability and reducing false alarms.

# Repository Structure
 - `config.py`: Contains configuration settings and constants used throughout the project, including default values for patche size, batch sizes, and number of epochs. Note that hyperparameters for evaluation are also defined separately within the hyperparameter tuning script

 - `data_preparation.py`: Includes functions and methods for reading, preprocessing, and splitting the data into training, validation, and test sets. It also handles patch creation and data normalization.

- `visualization.py`: Provides functions for visualizing various aspects of the data and model performance, including plotting training and validation metrics, DEMs, and evaluation metrics.

- `cnn_unet.py`: Contains the implementation of the Convolutional Neural Network (CNN) U-Net model used for improving the DEM resolution. It also includes loss functions and evaluation metrics.

- `hyperparameter_tuning.py`: Defines the required hyperparameter tuning for our model, which will be implemented simultaneously during model training by running training.py.

- `training.py`: Integrates model training and hyperparameter tuning processes. Executes the training of the CNN U-Net model and performs hyperparameter optimization.
  
- `testing.py`: Evaluates the trained model on test data, calculates performance metrics, and generates reconstructed output DEMs.

- `evaluation.py`: Performs detailed evaluation and comparison of the enhanced DEM against reference DEM and GDEMs, including statistical analysis and visualization of results.

# Get Started
To set up and run the scripts, follow these steps:

1. Install Required Packages:
Install the required packages listed in the requirements.txt file:
`pip install -r requirements.txt`

2. Prepare the Input Data:
After performing the necessary preprocessing for remote imagery data using ArcGIS Pro and SNAP software, stack all processed bands and indices into a single raster layer in the following order: GDEMs, Landsat 8 bands, indices from Landsat 8, Sentinel-2 bands, indices from Sentinel-2, Sentinel-1 bands, OSM layers, and the target 5-meter DEM.

3. Set Configuration:
Ensure all configuration settings and constants are correctly defined in config.py, including file paths, hyperparameters, and other key settings.

4. Run Training and Hyperparameter Tuning:
Execute the training.py script to train the model and perform hyperparameter tuning simultaneously:
`python training.py`

5. Run Model Over Test Area:
Use the testing.py script to evaluate the trained model on the test area, calculate performance metrics, and generate reconstructed output DEMs:
`python testing.py`

6. Detailed Evaluation:
Run the evaluation.py script for a detailed evaluation and comparison of the enhanced DEM against reference DEM and GDEMs, including statistical analysis and visualization of results:
`python evaluation.py`

# Requirements
To ensure all scripts run correctly, the following Python packages must be installed:

`numpy`: For numerical operations and array manipulation.

`torch`: For building and training the CNN U-Net model.

`rasterio`: For reading and writing raster data.

`geopanda`: For handling geospatial data.

`matplotlib`: For plotting and visualizations.

`optuna`: For hyperparameter tuning.

`tqdm`: For displaying progress bars during training and evaluation.

`scikit-learn`: For calculating evaluation metrics.


# Citation
If you use this code or the framework in your research, please cite our paper:

Zandsalimi, Z., Barbosa, S.A., Goodall, J.L., Alemazkoor, N., Shafiee-Jood, M. (2024). Deep Learning-Based Downscaling of Global Digital Elevation Models for Enhanced Urban Flood Modeling. [Accepted to the Journal of Hydrology].
