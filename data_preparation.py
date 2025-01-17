# data_handling.py

import torch
import random
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset

# Set seeds for reproducibility
def set_seed(random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

# Function to check if CUDA (GPU support) is available
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Call the set_seed function to set the random seed
random_state = 42
set_seed(random_state)

# Check if CUDA (GPU support) is available
device = get_device()
print("Using device:", device)

# Function to plot the bands of the raster data
def plot_bands(raster_data):
    num_bands = len(raster_data)
    bands_per_row = 6
    for i in range(0, num_bands, bands_per_row):
        plt.figure(figsize=(20, 4))  # Adjust the figure size as needed
        for j in range(bands_per_row):
            if i + j < num_bands:
                plt.subplot(1, bands_per_row, j + 1)
                plt.imshow(raster_data[i + j], cmap='gray')
                plt.title(f'Band {i + j + 1}')
                plt.colorbar()
        plt.show()
        
# Function to read and preprocess the data
def read_and_preprocess_data(bip_file_path, NoData=-9999):
    with rasterio.open(bip_file_path) as src:
        raster_data = src.read()
        last_band = src.read(45)  # 45: last band: Target DEM 5m
        original_width, original_height = src.width, src.height
        original_transform = src.transform
        input_crs = src.crs

        # Mask the NoData values
        raster_data = np.ma.masked_where(raster_data == NoData, raster_data)
        
    print("Original shape of raster_data:", raster_data.shape)
    print("Original transform:", original_transform)
    print("Input layer CRS:", input_crs)
    
    return raster_data, last_band, original_transform, input_crs

# Function to load the shapefile
def load_shapefile(shapefile_path):
    return gpd.read_file(shapefile_path)

# ===================================================================
# ============================ Cropping =============================
# ===================================================================
# Cropping the original input based on patch size
def crop_for_patching(data, patch_size):
    """
    Crop the data from each side equally to make its second and third dimensions 
    divisible by patch_size.

    Args:
    data: 3D numpy array, shape (channels, height, width)
    patch_size: Size of the square window (patch)

    Returns:
    Cropped data as a 3D numpy array
    """
    _, height, width = data.shape

    # Calculate the excess pixels that prevent the image from being divisible by the patch size
    excess_height = height % patch_size
    excess_width = width % patch_size

    # Calculate how much to crop from each side
    crop_top = excess_height // 2
    crop_bottom = excess_height - crop_top
    crop_left = excess_width // 2
    crop_right = excess_width - crop_left

    # Perform cropping
    cropped_data = data[:, crop_top:height-crop_bottom, crop_left:width-crop_right]

    return cropped_data, crop_left, crop_top

# Adjusting the Affine Transform After Cropping
def adjust_transform_for_cropping(original_transform, crop_left, crop_top, pixel_width, pixel_height):
    """
    Adjust the affine transform to account for cropping.

    Args:
    original_transform (Affine): The original affine transform of the raster.
    crop_left (int): Number of pixels cropped from the left.
    crop_top (int): Number of pixels cropped from the top.
    pixel_width (float): Pixel size in x-direction (geographic units per pixel).
    pixel_height (float): Pixel size in y-direction (geographic units per pixel, positive value).

    Returns:
    Affine: The adjusted affine transform.
    """
    c, f = original_transform.c, original_transform.f
    new_c = c + (crop_left * pixel_width)
    # Subtract for crop_top because the y-axis increases downwards
    new_f = f - (crop_top * pixel_height)
    return original_transform._replace(c=new_c, f=new_f)

# ===================================================================
# ===================== Defined Bands' Headings =====================
# ===================================================================
# Define band headings
band_headings = {
    # Global Digital Elevation Models (GDEMs)
    0: "ALOS30m (GDEM)",
    1: "SRTM30m (GDEM)",
    2: "NASADEM30m (GDEM)",
    3: "ASTER30m (GDEM)",
    # Landsat 8 OLI # Operational Land Imager (OLI) >>> Nine spectral bands, including a pan band:
    4: "LC08_B01", # Coastal Aerosol (0.43 - 0.45 µm) 30 m
    5: "LC08_B02", # Blue (0.450 - 0.51 µm) 30 m
    6: "LC08_B03", # Green (0.53 - 0.59 µm) 30 m
    7: "LC08_B04", # Red (0.64 - 0.67 µm) 30 m
    8: "LC08_B05", # Near-Infrared (0.85 - 0.88 µm) 30 m
    9: "LC08_B06", # SWIR 1(1.57 - 1.65 µm) 30 m
    10: "LC08_B07", # SWIR 2 (2.11 - 2.29 µm) 30 m
    11: "LC08_B08", # Panchromatic (PAN) (0.50 - 0.68 µm) 15 m
    12: "LC08_B09", # Cirrus (1.36 - 1.38 µm) 30 m >>> remove
                    # Thermal Infrared Sensor (TIRS)
    13: "LC08_B10", # Band 10 TIRS 1 (10.6 - 11.19 µm) 100 m >>> remove
    14: "LC08_B11", # Band 11 TIRS 2 (11.5 - 12.51 µm) 100 m >>> remove
    15: "LC08_NDWI",
    16: "LC08_NDVI",
    17: "LC08_NDBI",
    18: "LC08_BSI",
    19: "LC08_EVI",
    20: "LC08_AWEI",
    # Sentinel 2 # 
    21: "S02_B01", # Coastal aerosol (0.443 µm) 60 m
    22: "S02_B02", # Blue (0.490 µm) 10 m
    23: "S02_B03", # Green (0.560 µm) 10 m
    24: "S02_B04", # Red (0.665 µm) 10 m
    25: "S02_B05", # Vegetation red edge (0.705 µm) 20 m
    26: "S02_B06", # Vegetation red edge (0.740 µm) 20 m
    27: "S02_B07", # Vegetation red edge (0.783 µm) 20 m
    28: "S02_B08", # NIR (0.842 µm) 10 m
    29: "S02_B8A", # Vegetation red edge (0.865 µm) 20 m
    30: "S02_B09", # Water vapour (0.945 µm) 60 m >>> remove
    31: "S02_B10", # SWIR - Cirrius (1.375 µm) 60 m >>> remove
    32: "S02_B11", # SWIR (1.610 µm) 20 m
    33: "S02_B12", # SWIR (2.190 µm) 20 m
    34: "S02_NDWI",
    35: "S02_NDVI",
    36: "S02_NDBI",
    37: "S02_BSI",
    38: "S02_EVI",
    39: "S02_AWEI",
    # Sentinel 1
    40: "S01_Gamma0_VH",
    41: "S01_Gamma0_VV",
    # Open Street Map
    42: "Building_OSM",
    43: "Road_OSM",
    # Target value
    44: "Target_DEM5m"
}

# Example usage: print the heading for band 1
#print("Heading for Band 1:", band_headings[0])

# ===================================================================
# ========================== Band Selecting =========================
# ===================================================================
def select_bands(raster_data, band_indices):
    """Select specific bands from the masked data based on the provided indices."""
    selected_bands = [raster_data[i] for i in band_indices]
    return selected_bands

# Define the indices of the bands you want to select
# GDEMs, Landsat 8 (without thermal bands), OSM, DEM 5m
required_band_indices_AllData = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44] # Update these indices based on your needs 

# ===================================================================
# ========================== Normalization ==========================
# ===================================================================
def normalize_band(band, normalization='standardization'):
    """Normalize a single band."""
    if normalization == 'min_max':
        min_val = band.min()
        max_val = band.max()
        # Avoid division by zero
        return (band - min_val) / (max_val - min_val) if max_val != min_val else band
    elif normalization == 'standardization':
        mean_val = band.mean()
        std_val = band.std()
        # Avoid division by zero
        return (band - mean_val) / std_val if std_val != 0 else band
    else:
        return band  # If no normalization method is specified
    
def normalize_selected_bands(selected_bands, normalization='min_max'):
    """Normalize all selected bands except the penultimate one."""
    normalized_bands = []
    num_bands = len(selected_bands)
    for i in range(num_bands):
        band = selected_bands[i]
        # Skip normalization for the penultimate band
        if i not in (num_bands - 3, num_bands - 2):  # Adjusted based on 0-based indexing
            band = normalize_band(band, normalization)
        normalized_bands.append(band)
    return normalized_bands

# ===================================================================
# ======================== Reshaping for CNN ========================
# ===================================================================
def reshape_for_cnn(normalized_selected_bands):
    """Reshape the selected bands to form a 3D array suitable for CNN input."""
    return np.stack(normalized_selected_bands, axis=-1)

# ===================================================================
# =========================== Patching ==============================
# ===================================================================
def create_patches(data, window_size, overlap_percentage, transform):
    """
    Create patches from the data with specified window size and adjusted step size to ensure coverage,
    including specific handling for the last patches on the right and bottom to cover edges.

    Args:
    data: 3D numpy array, shape (height, width, channels)
    window_size: Size of each patch
    overlap_percentage: Overlap percentage between patches

    Returns:
    Patches as a 4D numpy array
    """
    step_size = int(window_size * (1 - overlap_percentage))
    
    # Calculate the number of patches that can fit along each dimension
    num_patches_x = int(np.ceil((data.shape[1] - window_size) / step_size)) + 1
    num_patches_y = int(np.ceil((data.shape[0] - window_size) / step_size)) + 1

    patches = []
    bbox_coordinates = []

    for i in range(num_patches_y):
        y_start = i * step_size
        for j in range(num_patches_x):
            x_start = j * step_size

            # Adjust for the last patch in each row/column to ensure it covers to the edge
            if x_start + window_size > data.shape[1]:
                x_start = data.shape[1] - window_size
            if y_start + window_size > data.shape[0]:
                y_start = data.shape[0] - window_size

            patch = data[y_start:y_start + window_size, x_start:x_start + window_size, :]
            patches.append(patch)

            # Calculate geographic coordinates for the bounding box
            top_left_geo = transform * (x_start, y_start)
            top_right_geo = transform * (x_start + window_size, y_start)
            bottom_left_geo = transform * (x_start, y_start + window_size)
            bottom_right_geo = transform * (x_start + window_size, y_start + window_size)
            bbox_coordinates.append((top_left_geo, top_right_geo, bottom_left_geo, bottom_right_geo))

    return np.array(patches), bbox_coordinates, num_patches_x, num_patches_y

def get_indices_from_subset(dataset_subset):
    return [dataset_subset.indices[i] for i in range(len(dataset_subset))]

def prepared_and_split_data(patch_size, raster_data, overlap_percentage, batch_size, original_transform, test_area_gdf, random_state=42):
    """
    Prepares spatial data and splits it into training, validation, and test DataLoader objects.

    Parameters: 
    - patch_size: The size of each patch to be created.
    - raster_data: The input raster data.
    - overlap_percentage: The percentage of overlap between adjacent patches.
    - batch_size: The size of each batch for DataLoader.
    - original_transform: The original affine transform of the raster.
    - test_area_gdf: The GeoDataFrame containing the test area shapefile.
    - random_state: Seed for reproducibility.
    
    Returns:
    - train_loader: DataLoader for the training set.
    - validation_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    - features, features_patches, target_patches, num_patches_x, num_patches_y, bbox_coordinates, cropped_raster_data, train_indices, val_indices, test_patches_idx, new_transform
    """
    
    # Set seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
   
    # Step 1: Crop raster data for patching
    cropped_raster_data, crop_left, crop_top = crop_for_patching(raster_data, patch_size)

    # Adjusting the Affine Transform After Cropping
    new_transform = adjust_transform_for_cropping(
        original_transform,
        crop_left=crop_left,
        crop_top=crop_top,
        pixel_width=original_transform.a,  # Pixel size in x-direction
        pixel_height=abs(original_transform.e)  # Pixel size in y-direction (abs because e is often negative)
    )

    # Apply fixed band selection to the cropped data
    selected_bands = select_bands(cropped_raster_data, required_band_indices_AllData) # bands are selected by user (different combination of data)

    # Normalization
    normalization_method = 'min_max'  # or 'standardization'
    normalized_selected_bands = normalize_selected_bands(selected_bands, normalization=normalization_method)

    # Reshape for CNN
    cnn_input_data = reshape_for_cnn(normalized_selected_bands)
    features = cnn_input_data[:, :, :-1]  # All bands except the last one
    target = cnn_input_data[:, :, -1:]   # The last band

    # Patching
    window_size = patch_size  # Patch size

    features_patches, bbox_coordinates, num_patches_x, num_patches_y = create_patches(features, window_size, overlap_percentage, new_transform)
    target_patches, bbox_coordinates, num_patches_x, num_patches_y = create_patches(target, window_size, overlap_percentage, new_transform)

    # Conversion to PyTorch tensors and rearranging
    features_patches_tensor = torch.from_numpy(features_patches).float().permute(0, 3, 1, 2)
    target_patches_tensor = torch.from_numpy(target_patches).float().squeeze(-1).unsqueeze(1)

    # Create Polygon geometries from bounding box coordinates
    patch_polygons = [Polygon([bbox[0], bbox[2], bbox[3], bbox[1], bbox[0]]) for bbox in bbox_coordinates]

    # Create a GeoDataFrame from these geometries
    patches_gdf = gpd.GeoDataFrame(geometry=patch_polygons, crs=test_area_gdf.crs)  # Ensure CRS matches your shapefile
    
    # Union of all geometries in the test_area_gdf if there are multiple
    test_area_union = test_area_gdf.unary_union
    test_patches_idx = patches_gdf.index[patches_gdf.intersects(test_area_union)].tolist()

    # Create the test dataset using Subset
    combined_dataset = TensorDataset(features_patches_tensor, target_patches_tensor)
    test_dataset = Subset(combined_dataset, test_patches_idx)

    # Indices for all patches
    all_indices = set(range(len(combined_dataset)))

    # Indices for remaining patches (training + validation)
    remaining_indices = list(all_indices - set(test_patches_idx))

    # Split remaining indices into training and validation
    train_indices, val_indices = train_test_split(remaining_indices, test_size=0.2, random_state=random_state)  # Adjust the test_size as needed

    # Create training and validation datasets using Subset
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    # Create DataLoader for each dataset part
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader, features, features_patches, target_patches, num_patches_x, num_patches_y, bbox_coordinates, cropped_raster_data, train_indices, val_indices, test_patches_idx, new_transform
