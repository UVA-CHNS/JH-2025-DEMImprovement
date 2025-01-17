import os

# config.py

# Paths

# Path to the input raster data for training. This is a stacked raster layer that includes GDEMs, Landsat 8, 
# Indices of LC08, Sentinel-2, Indices of S02, Sentinel-1, OSM layers, and the Target DEM, all at a spatial 
# resolution of 5m. The data is in BIP format and was extracted using ArcGIS Pro after required preprocessing.
BIP_FILE_PATH = r'E:\Portsmouth city\Final Imagery\Final_Data07May2024\For CNN model\AllData_5m.bip'

# Path to the input raster data for testing. Similar to BIP_FILE_PATH but specifically for the test area.
BIP_FILE_PATH_TEST = r'E:\Portsmouth city\Final Imagery\Final_Data07May2024\For CNN model\AllData_5m_TestArea.bip'

# Path to the shapefile of the test area (rectangular). This shapefile was used for data splitting into training, 
# validation, and test sets.
SHAPEFILE_PATH = r'E:\Portsmouth city\Shapefiles\TUFLOW_Area\CNN_Area\TestArea.shp'

# Path to the shapefile of the flooded area within the test area. The final improved DEM was evaluated for this 
# specific area in terms of flood modeling and error statistics.
SHAPEFILE_PATH_TEST = r'E:\Portsmouth city\Shapefiles\TUFLOW_Area\New Area\FloodedArea.shp'

# Base directory to save all outputs.
BASE_OUTPUT_DIR = r'E:\Portsmouth city\UNet Model Analysis\Output'

# Best model filename
BEST_MODEL_FILENAME = 'Test.pt'

# Subdirectories within the base output directory
BEST_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, 'best_model', BEST_MODEL_FILENAME)
OUTPUT_DIRECTORY_PLOTS = os.path.join(BASE_OUTPUT_DIR, 'plots')
OUTPUT_DIRECTORY_MAPS = os.path.join(BASE_OUTPUT_DIR, 'maps_tiff')

# Data Handling
# Size of the patches used for training and testing the model.
PATCH_SIZE = 128

# Percentage of overlap between patches.
OVERLAP_PERCENTAGE = 0.1

# Batch size used for training and testing the model.
BATCH_SIZE = 64

# Value representing no data in the input raster layers.
NO_DATA_VALUE = -9999

# Training
# Number of epochs to train the model.
NUM_EPOCHS = 10
