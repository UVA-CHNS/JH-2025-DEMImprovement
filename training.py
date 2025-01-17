import torch
import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show
from data_preparation import read_and_preprocess_data, load_shapefile, plot_bands, set_seed, prepared_and_split_data
from visualization import plot_raster_with_colorbar, plot_raster_and_shapefile, plot_map_with_patches, save_and_close_plot
from hyperparameter_tuning import run_hyperparameter_tuning
from config import PATCH_SIZE, OVERLAP_PERCENTAGE, BATCH_SIZE, BEST_MODEL_PATH, NUM_EPOCHS, NO_DATA_VALUE, BIP_FILE_PATH, SHAPEFILE_PATH, SHAPEFILE_PATH_TEST, OUTPUT_DIRECTORY_PLOTS

# Set seeds for reproducibility
random_state = 42
set_seed(random_state)

# Define paths
bip_file_path = BIP_FILE_PATH
shapefile_path = SHAPEFILE_PATH

# Read and preprocess the data
NoData = NO_DATA_VALUE
raster_data, last_band, original_transform, input_crs = read_and_preprocess_data(bip_file_path, NoData)

# Plot the raster with colorbar
fig, ax = plot_raster_with_colorbar(last_band, NoData, original_transform)
save_and_close_plot(os.path.join(OUTPUT_DIRECTORY_PLOTS, 'raster_colorbar.png'), fig)

# Load the shapefile
test_area_gdf = load_shapefile(shapefile_path)

# Plot the raster and shapefile together
fig, ax = plot_raster_and_shapefile(raster_data, last_band, NoData, original_transform, test_area_gdf)
save_and_close_plot(os.path.join(OUTPUT_DIRECTORY_PLOTS, 'raster_shapefile.png'), fig)

# Prepare and split data
patch_size = PATCH_SIZE
overlap_percentage = OVERLAP_PERCENTAGE
batch_size = BATCH_SIZE
train_loader, validation_loader, test_loader, features, features_patches, target_patches, num_patches_x, num_patches_y, bbox_coordinates, cropped_raster_data, train_indices, val_indices, test_patches_idx, new_transform = prepared_and_split_data(
    patch_size, raster_data, overlap_percentage, batch_size, original_transform, test_area_gdf, random_state
)

# Printing the bounding box coordinates of the first few patches for inspection
for i, bbox in enumerate(bbox_coordinates[:5]):  # Adjust the slice for more or fewer patches
    print(f"Patch {i+1} Bounding Box Coordinates: {bbox}")

# Plot map with patches
fig, ax = plot_map_with_patches(cropped_raster_data, bbox_coordinates, train_indices, val_indices, test_patches_idx, new_transform)
save_and_close_plot(os.path.join(OUTPUT_DIRECTORY_PLOTS, 'map_with_patches.png'), fig)

# Run hyperparameter tuning
run_hyperparameter_tuning(raster_data, original_transform, test_area_gdf)
