import torch
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show
from torch.utils.data import DataLoader, TensorDataset
from visualization import plot_map_with_test_patches, save_and_close_plot
from cnn_unet import UNet, calculate_rmse, calculate_mae, calculate_mse  # Import your U-Net model definition
from data_preparation import (
    read_and_preprocess_data, load_shapefile, crop_for_patching, adjust_transform_for_cropping, 
    select_bands, normalize_selected_bands, reshape_for_cnn, create_patches, required_band_indices_AllData
)
from config import PATCH_SIZE, OVERLAP_PERCENTAGE, BATCH_SIZE, BEST_MODEL_PATH, NO_DATA_VALUE, BIP_FILE_PATH_TEST, OUTPUT_DIRECTORY_PLOTS, OUTPUT_DIRECTORY_MAPS, BEST_MODEL_FILENAME

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create the directory if it does not exist
if not os.path.exists(OUTPUT_DIRECTORY_PLOTS):
    os.makedirs(OUTPUT_DIRECTORY_PLOTS)

# ===================================================================
# =========== Model Prediction using the best parameters ============
# ===================================================================
# Assuming `best_params` is a dictionary containing all the optimized parameters:
best_params = {
    'in_channels': 39,
    'out_channels': 1,
    'depth': 4,
    'base_filters': 64,
    'kernel_size': 3,
    'pool_size': 2,
    'dropout_rate': 0,
    'use_dropout_down': False,
    'use_dropout_bottleneck': True,
    'use_dropout_up': False
}

# Model setup
best_model = UNet(
    in_channels=best_params['in_channels'],
    out_channels=best_params['out_channels'],
    depth=best_params['depth'],
    base_filters=best_params['base_filters'],
    kernel_size=best_params['kernel_size'],
    pool_size=best_params['pool_size'],
    dropout_rate=best_params['dropout_rate'],
    use_dropout_down=best_params['use_dropout_down'],
    use_dropout_bottleneck=best_params['use_dropout_bottleneck'],
    use_dropout_up=best_params['use_dropout_up']
).to(device)

# Load the best model weights
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()  # Set the model to evaluation mode

# ===================================================================
# ======================= Prediction_Test Data ======================
# ===================================================================
# Function to calculate metrics
def calculate_metrics(test_loader, best_model, device):
    best_model.eval()
    total_rmse, total_mae, total_mse = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)

            batch_rmse = calculate_rmse(outputs, targets, no_data_val=-9999)
            batch_mae = calculate_mae(outputs, targets, no_data_val=-9999)
            batch_mse = calculate_mse(outputs, targets, no_data_val=-9999)

            total_rmse += batch_rmse * inputs.size(0)
            total_mae += batch_mae * inputs.size(0)
            total_mse += batch_mse * inputs.size(0)
            total_samples += inputs.size(0)

    avg_rmse = total_rmse / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    print(f'Average RMSE on Test Set: {avg_rmse}')
    print(f'Average MAE on Test Set: {avg_mae}')
    print(f'Average MSE on Test Set: {avg_mse}')

# Load Test data
bip_file_path = BIP_FILE_PATH_TEST
NoData = NO_DATA_VALUE

with rasterio.open(bip_file_path) as src:
    test_data = src.read()
    test_transform = src.transform
    test_crs = src.crs
    test_data = np.ma.masked_where(test_data == NoData, test_data)

print("Original shape of raster_data:", test_data.shape)

# Export the necessary variables
__all__ = ['test_data', 'test_transform', 'test_crs']

# Preprocess the test data
patch_size = PATCH_SIZE
overlap_percentage = OVERLAP_PERCENTAGE
cropped_raster_data, crop_left, crop_top = crop_for_patching(test_data, patch_size)
new_transform = adjust_transform_for_cropping(test_transform, crop_left, crop_top, pixel_width=test_transform.a, pixel_height=abs(test_transform.e))
selected_bands = select_bands(cropped_raster_data, required_band_indices_AllData)
normalized_selected_bands = normalize_selected_bands(selected_bands, normalization='min_max')
cnn_input_data = reshape_for_cnn(normalized_selected_bands)
features = cnn_input_data[:, :, :-1]
target = cnn_input_data[:, :, -1:]

features_patches, bbox_coordinates, num_patches_x, num_patches_y = create_patches(features, patch_size, overlap_percentage, new_transform)
target_patches, bbox_coordinates, num_patches_x, num_patches_y = create_patches(target, patch_size, overlap_percentage, new_transform)

features_patches_tensor = torch.from_numpy(features_patches).float().permute(0, 3, 1, 2)
target_patches_tensor = torch.from_numpy(target_patches).float().squeeze(-1).unsqueeze(1)
combined_dataset = TensorDataset(features_patches_tensor, target_patches_tensor)
test_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate metrics
calculate_metrics(test_loader, best_model, device)


# Plot test area with test patches
fig, ax = plot_map_with_test_patches(cropped_raster_data, bbox_coordinates, new_transform, OUTPUT_DIRECTORY_PLOTS)
save_and_close_plot(os.path.join(OUTPUT_DIRECTORY_PLOTS, 'test_area_patches.png'), fig)

# ===================================================================
# =================== Reconstruct and Evaluate Output ===============
# ===================================================================
def reconstruct_from_patches(patches, original_shape, window_size, overlap_percentage):
    step_size = int(window_size * (1 - overlap_percentage))
    reconstructed = np.zeros(original_shape[:2])
    count_matrix = np.zeros(original_shape[:2])
    
    idx = 0
    num_patches_x = int(np.ceil((original_shape[1] - window_size) / step_size)) + 1
    num_patches_y = int(np.ceil((original_shape[0] - window_size) / step_size)) + 1

    for i in range(num_patches_y):
        y_start = i * step_size if i < num_patches_y - 1 else original_shape[0] - window_size
        for j in range(num_patches_x):
            x_start = j * step_size if j < num_patches_x - 1 else original_shape[1] - window_size

            patch = np.squeeze(patches[idx])
            if patch.ndim > 2:
                raise ValueError("Patch has more than 2 dimensions. Ensure patches are 2D.")
            
            reconstructed[y_start:y_start + window_size, x_start:x_start + window_size] += patch
            count_matrix[y_start:y_start + window_size, x_start:x_start + window_size] += 1
            idx += 1

    reconstructed /= np.maximum(count_matrix, 1)
    return reconstructed

# Example usage
test_predictions = []
best_model.eval()
with torch.no_grad():
    for patch in features_patches:
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        patch_tensor = patch_tensor.permute(0, 3, 1, 2)
        patch_tensor = patch_tensor.to(device)
        prediction = best_model(patch_tensor)
        test_predictions.append(prediction.cpu().numpy().squeeze(0))

original_height, original_width = features.shape[:2]
reconstructed_output = reconstruct_from_patches(test_predictions, (original_height, original_width), patch_size, 0.1)

# Add cropped pixels back to the reconstructed output
def calculate_cropping_amounts(original_height, original_width, patch_size):
    excess_height = original_height % patch_size
    excess_width = original_width % patch_size
    crop_top = excess_height // 2
    crop_bottom = excess_height - crop_top
    crop_left = excess_width // 2
    crop_right = excess_width - crop_left
    return crop_top, crop_bottom, crop_left, crop_right

def add_cropped_pixels(data, original_shape, crop_top, crop_bottom, crop_left, crop_right):
    _, original_height, original_width = original_shape
    padded_data = np.zeros((original_height, original_width))

    start_row = crop_top
    end_row = original_height - crop_bottom
    start_col = crop_left
    end_col = original_width - crop_right

    padded_data[start_row:end_row, start_col:end_col] = data
    return padded_data

# Original shape of raster data
original_shape = test_data.shape

# Calculate cropping amounts
crop_top, crop_bottom, crop_left, crop_right = calculate_cropping_amounts(original_shape[1], original_shape[2], patch_size)

# Add cropped pixels back to the reconstructed output
added_pixels_output = add_cropped_pixels(reconstructed_output, original_shape, crop_top, crop_bottom, crop_left, crop_right)

# Plotting the outputs
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot for reconstructed output
im0 = axs[0].imshow(reconstructed_output, cmap='gray')
axs[0].set_title('Reconstructed Output')
cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.03, pad=0.04)
cbar0.ax.set_ylabel('Normalized Elevation', rotation=270, labelpad=10)
cbar0.ax.tick_params(labelsize=8)

# Plot for added pixels output
im2 = axs[1].imshow(added_pixels_output, cmap='gray')
axs[1].set_title('Added Pixels Output')
cbar2 = fig.colorbar(im2, ax=axs[1], fraction=0.03, pad=0.04)
cbar2.ax.set_ylabel('Normalized Elevation', rotation=270, labelpad=10)
cbar2.ax.tick_params(labelsize=8)

plt.tight_layout()
save_and_close_plot(os.path.join(OUTPUT_DIRECTORY_PLOTS, 'reconstructed_output.png'), fig)

# ===================================================================
# === Exporting the final output (Improved DEM) in tiff format ======
# ===================================================================
# Assuming added_pixels_output contains the final reconstructed data
original_dem = added_pixels_output  # Replace with your actual final output variable name

# Reshape original_dem to add the channel dimension
original_dem_3d = original_dem[np.newaxis, :, :]  # Reshape to (1, 1105, 803)

# Create the directory if it does not exist
if not os.path.exists(OUTPUT_DIRECTORY_MAPS):
    os.makedirs(OUTPUT_DIRECTORY_MAPS)

# Define the path for the output DEM file within the new directory
output_dem_path = os.path.join(OUTPUT_DIRECTORY_MAPS, 'Improved_DEM_with_Padding.tif')  #'HyperTuning_Test1_3.tif'

# Exporting the Improved DEM with Original Transform and CRS
new_dem_meta = {
    'driver': 'GTiff',
    'height': original_dem_3d.shape[1],
    'width': original_dem_3d.shape[2],
    'count': original_dem_3d.shape[0],  # 1, since it's a single-channel image
    'dtype': original_dem_3d.dtype,
    'crs': test_crs,
    'transform': test_transform,
    'nodata': NO_DATA_VALUE  # Specify the no-data value
}

# Export the DEM using rasterio
with rasterio.open(output_dem_path, 'w', **new_dem_meta) as dst:
    dst.write(original_dem_3d[0, :, :], 1)  # Write the single channel

print(f"Raster file saved to {output_dem_path}")
print("Done")