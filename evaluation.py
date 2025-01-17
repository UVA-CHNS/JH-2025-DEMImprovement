# analyze_dem.py
import os
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import geopandas as gpd
from testing import cropped_raster_data, reconstructed_output, new_transform, test_transform, test_crs
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import OUTPUT_DIRECTORY_PLOTS, SHAPEFILE_PATH_TEST, NO_DATA_VALUE, OUTPUT_DIRECTORY_MAPS
from visualization import save_and_close_plot, plot_evaluation_metrics    # Importing the function from visualization


# Define the output directory
output_dir = OUTPUT_DIRECTORY_PLOTS

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# GDEM_ALOS: one of the low resolution DEMs for comparison with the high resolution DEM-5m
GDEM_ALOS = cropped_raster_data[0, :, :]  # Assuming the first dimension is bands
print("shape of ALOS:", GDEM_ALOS.shape)

# GDEM_SRTM: one of the low resolution DEMs for comparison with the high resolution DEM-5m
GDEM_SRTM = cropped_raster_data[1, :, :]  # Assuming the first dimension is bands
print("shape of SRTM:", GDEM_SRTM.shape)

# GDEM_NASADEM: one of the low resolution DEMs for comparison with the high resolution DEM-5m
GDEM_NASADEM = cropped_raster_data[2, :, :]  # Assuming the first dimension is bands
print("shape of NASADEM:", GDEM_NASADEM.shape)

# GDEM_ASTER: one of the low resolution DEMs for comparison with the high resolution DEM-5m
GDEM_ASTER = cropped_raster_data[3, :, :]  # Assuming the first dimension is bands
print("shape of ASTER:", GDEM_ASTER.shape)

# Target DEM as high resolution map
Target_DEM5m = cropped_raster_data[44, :, :]
print("shape of DEM5m:", Target_DEM5m.shape)

# Error Function (Improved DEM - Target DEM)
def calculate_difference(map1, map2):
    # Assuming map1 and map2 are numpy arrays of the same shape
    # Map1: Improved DEM
    # Map2: Target DEM
    return np.abs(map1 - map2)

# Plots
mask = Target_DEM5m == NO_DATA_VALUE

fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns

# Plot 1: Main Target Map after Cropping
im0 = axs[0, 0].imshow(Target_DEM5m, cmap='gray')
axs[0, 0].set_title('Target Map (High resolution)')
fig.colorbar(im0, ax=axs[0, 0])

# Plot 2: ALOS 30m
im1 = axs[0, 1].imshow(GDEM_ALOS, cmap='gray')
axs[0, 1].set_title('ALOS 30m')
fig.colorbar(im1, ax=axs[0, 1])

# Plot 3: Improved DEM
max_value_last_channel = np.max(cropped_raster_data[-1, :, :])
min_value_last_channel = np.min(cropped_raster_data[-1, :, :])
improved_DEM = (reconstructed_output * (max_value_last_channel - min_value_last_channel)) + min_value_last_channel
improved_DEM_masked = np.ma.array(improved_DEM, mask=mask)
im2 = axs[1, 0].imshow(improved_DEM_masked, cmap='gray')
axs[1, 0].set_title('Improved DEM')
fig.colorbar(im2, ax=axs[1, 0])

# Plot 4: Difference Map
difference_map = calculate_difference(Target_DEM5m, improved_DEM)
im3 = axs[1, 1].imshow(difference_map, cmap='gray')
axs[1, 1].set_title('Difference Map (Improved DEM - Target DEM)')
fig.colorbar(im3, ax=axs[1, 1])

plt.tight_layout()
save_and_close_plot(os.path.join(output_dir, 'comparison_plots.png'), fig)  # Save and close the figure

# Save the Improved DEM as a GeoTIFF
output_path = os.path.join(OUTPUT_DIRECTORY_MAPS, 'Improved_DEM.tif')

# Save the raster
with rasterio.open(
    output_path, 'w', driver='GTiff',
    height=improved_DEM.shape[0],
    width=improved_DEM.shape[1],
    count=1,  # number of raster bands
    dtype=improved_DEM.dtype,
    crs=test_crs,
    transform=new_transform
) as dst:
    dst.write(improved_DEM, 1)

print('Improved DEM saved to', output_path)

# Load the shapefile
area_of_interest = gpd.read_file(SHAPEFILE_PATH_TEST)

# Plot the shapefile
fig, ax = plt.subplots(figsize=(10, 10))  # Create a figure and an axes.
area_of_interest.plot(ax=ax, color='blue')  # Plot the area of interest with blue color
ax.set_title('Area of Interest - Shapefile Plot')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# plt.savefig(os.path.join(output_dir, 'shapefile_plot.png'))  # Save the plot
# plt.show()  # Commented out to prevent interactive display

# Define the color map and normalization
cmap = cm.gray  # Using the colormap directly from cm (colormap) for consistency
norm = Normalize(vmin=Target_DEM5m.min(), vmax=Target_DEM5m.max())  # Normalizes data values

# Load the shapefile 
test_area_gdf = gpd.read_file(SHAPEFILE_PATH_TEST)

# Print the CRS
print("CRS (Coordinate Reference System):", test_area_gdf.crs)

fig, ax = plt.subplots(figsize=(8, 8))
cax = show(Target_DEM5m, ax=ax, transform=new_transform, cmap=cmap, norm=norm)

# Overlay the shapefile
test_area_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2)

# Optionally, re-show the colorbar and plot to ensure the overlay is properly rendered
fig.colorbar(cax.images[0], ax=ax, orientation='vertical', fraction=0.036, pad=0.04, label='Elevation (m)')
save_and_close_plot(os.path.join(output_dir, 'shapefile_overlay.png'), fig)  # Save and close the plot

# Function to apply mask to a raster data array
def mask_dem(dem_array, shapes, transform, crs):
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=dem_array.shape[0],
            width=dem_array.shape[1],
            count=1,
            dtype=dem_array.dtype,
            transform=transform,
            crs=crs
        ) as dataset:
            dataset.write(dem_array, 1)
            out_image, out_transform = rasterio_mask(dataset, shapes, crop=True)
            return out_image[0]  # Return the first band

# Mask each DEM
masked_GDEM_ALOS = mask_dem(GDEM_ALOS, area_of_interest.geometry, new_transform, test_crs)
masked_GDEM_SRTM = mask_dem(GDEM_SRTM, area_of_interest.geometry, new_transform, test_crs)
masked_GDEM_NASADEM = mask_dem(GDEM_NASADEM, area_of_interest.geometry, new_transform, test_crs)
masked_GDEM_ASTER = mask_dem(GDEM_ASTER, area_of_interest.geometry, new_transform, test_crs)
masked_Target_DEM5m = mask_dem(Target_DEM5m, area_of_interest.geometry, new_transform, test_crs)
masked_Improved_DEM5m = mask_dem(improved_DEM_masked, area_of_interest.geometry, new_transform, test_crs)

# Print shapes of the masked DEMs to confirm
print("Masked shape of ALOS:", masked_GDEM_ALOS.shape)
print("Masked shape of SRTM:", masked_GDEM_SRTM.shape)
print("Masked shape of NASADEM:", masked_GDEM_NASADEM.shape)
print("Masked shape of ASTER:", masked_GDEM_ASTER.shape)
print("Masked shape of DEM5m:", masked_Target_DEM5m.shape)
print("Masked shape of Improved DEM:", masked_Improved_DEM5m.shape)

flattened_alos = masked_GDEM_ALOS.flatten()
flattened_srtm = masked_GDEM_SRTM.flatten()
flattened_nasadem = masked_GDEM_NASADEM.flatten()
flattened_aster = masked_GDEM_ASTER.flatten()
flattened_target = masked_Target_DEM5m.flatten()
flattened_improved = masked_Improved_DEM5m.flatten()

mask = flattened_target != NO_DATA_VALUE

# Apply Masks to Each DEM and Target Array:
filtered_target = flattened_target[mask]
filtered_alos = flattened_alos[mask]
filtered_srtm = flattened_srtm[mask]
filtered_nasadem = flattened_nasadem[mask]
filtered_aster = flattened_aster[mask]
filtered_improved = flattened_improved[mask]

# Calculate metrics for ALOS vs Target
rmse_alos_target = np.sqrt(mean_squared_error(filtered_target, filtered_alos))
mae_alos_target = mean_absolute_error(filtered_target, filtered_alos)
r2_alos_target = r2_score(filtered_target, filtered_alos)

# Calculate metrics for SRTM vs Target
rmse_srtm_target = np.sqrt(mean_squared_error(filtered_target, filtered_srtm))
mae_srtm_target = mean_absolute_error(filtered_target, filtered_srtm)
r2_srtm_target = r2_score(filtered_target, filtered_srtm)

# Calculate metrics for NASADEM vs Target
rmse_nasadem_target = np.sqrt(mean_squared_error(filtered_target, filtered_nasadem))
mae_nasadem_target = mean_absolute_error(filtered_target, filtered_nasadem)
r2_nasadem_target = r2_score(filtered_target, filtered_nasadem)

# Calculate metrics for ASTER vs Target
rmse_aster_target = np.sqrt(mean_squared_error(filtered_target, filtered_aster))
mae_aster_target = mean_absolute_error(filtered_target, filtered_aster)
r2_aster_target = r2_score(filtered_target, filtered_aster)

# Calculate metrics for Improved vs Target
rmse_improved_target = np.sqrt(mean_squared_error(filtered_target, filtered_improved))
mae_improved_target = mean_absolute_error(filtered_target, filtered_improved)
r2_improved_target = r2_score(filtered_target, filtered_improved)

def calculate_mbe(y_true, y_pred):
    """Calculate Mean Bias Error."""
    return np.mean(y_pred - y_true)

# Calculate MBE for each comparison
mbe_alos_target = calculate_mbe(filtered_target, filtered_alos)
mbe_srtm_target = calculate_mbe(filtered_target, filtered_srtm)
mbe_nasadem_target = calculate_mbe(filtered_target, filtered_nasadem)
mbe_aster_target = calculate_mbe(filtered_target, filtered_aster)
mbe_improved_target = calculate_mbe(filtered_target, filtered_improved)

# Function to calculate linear regression line
def linear_regression_line(x, y):
    coefficients = np.polyfit(x, y, 1)  # 1st degree polynomial for linear regression
    polynomial = np.poly1d(coefficients)
    trendline = polynomial(x)
    return trendline

# Calculate trend lines
trendline_alos_target = linear_regression_line(filtered_alos, filtered_target)
trendline_srtm_target = linear_regression_line(filtered_srtm, filtered_target)
trendline_nasadem_target = linear_regression_line(filtered_nasadem, filtered_target)
trendline_aster_target = linear_regression_line(filtered_aster, filtered_target)
trendline_improved_target = linear_regression_line(filtered_improved, filtered_target)

# Histogram
ALOS_elevation = filtered_alos[filtered_alos != -9999]
SRTM_elevation = filtered_srtm[filtered_srtm != -9999]
NASADEM_elevation = filtered_nasadem[filtered_nasadem != -9999]
ASTER_elevation = filtered_aster[filtered_aster != -9999]
Target_elevation = filtered_target[filtered_target != -9999]
Improved_DEM_elevation = filtered_improved[filtered_improved != -9999]

# Plot evaluation metrics
plot_evaluation_metrics(filtered_alos, filtered_srtm, filtered_nasadem, filtered_aster, filtered_target, filtered_improved, 
                        trendline_alos_target, trendline_srtm_target, trendline_nasadem_target, trendline_aster_target, trendline_improved_target, 
                        rmse_alos_target, mae_alos_target, r2_alos_target, mbe_alos_target, 
                        rmse_srtm_target, mae_srtm_target, r2_srtm_target, mbe_srtm_target, 
                        rmse_nasadem_target, mae_nasadem_target, r2_nasadem_target, mbe_nasadem_target, 
                        rmse_aster_target, mae_aster_target, r2_aster_target, mbe_aster_target, 
                        rmse_improved_target, mae_improved_target, r2_improved_target, mbe_improved_target, output_dir)

print("Analysis and plotting completed. Outputs saved to:", output_dir)
print('Done')