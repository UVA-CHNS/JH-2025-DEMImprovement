# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from rasterio.plot import show
import matplotlib.patches as mpatches
import time

def save_and_close_plot(file_path, fig):
    fig.savefig(file_path)
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)

def plot_raster_with_colorbar(last_band, NoData, transform):
    # Mask the NoData values
    last_band_masked = np.ma.masked_where(last_band == NoData, last_band)

    # Define the color map and normalization
    cmap = cm.gray  # This is a commonly used color map, but you can choose others (e.g., 'gray', 'terrain', 'plasma')
    norm = Normalize(vmin=last_band_masked.min(), vmax=last_band_masked.max())  # Normalizes data values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the raster with the defined color map and normalization
    cax = show(last_band_masked, ax=ax, transform=transform, cmap=cmap, norm=norm)

    # Create a colorbar as a legend for the plot
    fig.colorbar(cax.images[0], ax=ax, orientation='vertical', fraction=0.036, pad=0.04, label='Elevation (m)')

    return fig, ax

def plot_raster_and_shapefile(raster_data, last_band, NoData, transform, shapefile_gdf):
    # Mask the NoData values
    last_band_masked = np.ma.masked_where(last_band == NoData, last_band)

    # Define the color map and normalization
    cmap = cm.gray  # This is a commonly used color map, but you can choose others (e.g., 'gray', 'terrain', 'plasma')
    norm = Normalize(vmin=last_band_masked.min(), vmax=last_band_masked.max())  # Normalizes data values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the raster with the defined color map and normalization
    cax = show(last_band_masked, ax=ax, transform=transform, cmap=cmap, norm=norm)

    # Create a colorbar as a legend for the plot
    fig.colorbar(cax.images[0], ax=ax, orientation='vertical', fraction=0.036, pad=0.04, label='Elevation (m)')

    # Overlay the shapefile
    shapefile_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2)

    return fig, ax

def create_patch_rectangle(bbox, color, label=None):
    # Assuming bbox has four points: top_left, top_right, bottom_right, bottom_left
    top_left, top_right, bottom_right, bottom_left = bbox
    width = top_right[0] - top_left[0]
    height = top_left[1] - bottom_left[1]
    rect = mpatches.Rectangle(top_left, width, height, linewidth=1, edgecolor=color, facecolor=color, alpha=0.3, label=label)
    return rect

def plot_map_with_patches(cropped_raster_data, bbox_coordinates, train_indices, val_indices, test_patches_idx, new_transform):
    # Extract the last band of cropped_raster_data as the target map
    target_map_data = cropped_raster_data[-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate the extent of the original raster for plotting
    height, width = target_map_data.shape
    left, top = new_transform * (0, 0)
    right, bottom = new_transform * (width, height)

    # Plot the target map
    ax.imshow(target_map_data, cmap='gray', extent=(left, right, bottom, top), interpolation='none')

    # Function to create a semi-transparent rectangle for a patch
    def create_patch_rectangle(patch_index, color, label=None):
        # Get the spatial coordinates for the corners of the patch
        top_left_geo, top_right_geo, bottom_right_geo, bottom_left_geo = bbox_coordinates[patch_index]
        # Calculate the width and height of the patch
        width = top_right_geo[0] - top_left_geo[0]
        height = top_left_geo[1] - bottom_left_geo[1]
        # Create and add the rectangle to the plot
        rect = mpatches.Rectangle((top_left_geo[0], bottom_left_geo[1]), width, height, linewidth=1, edgecolor='none', facecolor=color, alpha=0.5, label=label)
        ax.add_patch(rect)

    # Overlay the patches with semi-transparent rectangles
    for idx in train_indices:
        create_patch_rectangle(idx, 'green', label='Training' if idx == train_indices[0] else "")
    for idx in val_indices:
        create_patch_rectangle(idx, 'blue', label='Validation' if idx == val_indices[0] else "")
    for idx in test_patches_idx:
        create_patch_rectangle(idx, 'red', label='Testing' if idx == test_patches_idx[0] else "")

    # Create the legend and set the plot limits
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), prop={'size': 13, 'family': 'Times New Roman'})

    plt.xlabel('Longitude', fontsize=16, fontname='Times New Roman', fontweight='bold')
    plt.ylabel('Latitude', fontsize=16, fontname='Times New Roman', fontweight='bold')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    return fig, ax


# Visualization of trial metrics after each trial
def visualize_trial_metrics(trial_number, metrics, params, output_dir):
    epochs = range(1, len(metrics['train_loss']) + 1)

    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Formatting the parameters text
    params_text = "\n".join(f"{key}: {value}" for key, value in params.items())

    # Plot for Loss
    axs[0].plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')
    axs[0].plot(epochs, metrics['val_loss'], label='Validation Loss', color='green')
    axs[0].set_title(f'Trial {trial_number} - Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].text(0.5, -0.2, params_text, transform=axs[0].transAxes, ha='center', fontsize=9)
    axs[0].grid(True)

    # Plot for RMSE
    axs[1].plot(epochs, metrics['train_rmse'], label='Train RMSE', color='orange')
    axs[1].plot(epochs, metrics['val_rmse'], label='Validation RMSE', color='red')
    axs[1].set_title(f'Trial {trial_number} - RMSE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('RMSE')
    axs[1].legend()
    axs[1].text(0.5, -0.2, params_text, transform=axs[1].transAxes, ha='center', fontsize=9)
    axs[1].grid(True)

    # Plot for MAE
    axs[2].plot(epochs, metrics['train_mae'], label='Train MAE', color='purple')
    axs[2].plot(epochs, metrics['val_mae'], label='Validation MAE', color='brown')
    axs[2].set_title(f'Trial {trial_number} - MAE')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MAE')
    axs[2].legend()
    axs[2].text(0.5, -0.2, params_text, transform=axs[2].transAxes, ha='center', fontsize=9)
    axs[2].grid(True)

    plt.tight_layout()  # Adjust the layout
    save_and_close_plot(os.path.join(output_dir, f'trial_{trial_number}_metrics.png'), fig)

# Getting parameters of each trial
def get_params_for_trial(study, trial_number):
    for trial in study.trials:
        if trial.number == trial_number:
            return trial.params
    return None  # Return None or raise an exception if the trial number is not found

# Plotting Function for Best Parameters
def visualize_best_trial_metrics(trial, params, output_dir):
    # Extracting metrics from the trial
    metrics = trial.user_attrs['epoch_metrics']
    epochs = range(1, len(metrics['train_loss']) + 1)

    # Formatting the parameters text
    params_text = ", ".join(f"{key}: {value}" for key, value in params.items())

    fig, axs = plt.subplots(3, 1, figsize=(7, 14))

    # Subplot for Loss
    axs[0].plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')
    axs[0].plot(epochs, metrics['val_loss'], label='Validation Loss', color='green')
    axs[0].set_title(f'Trial {trial.number} - Loss (Params: {params_text})', fontsize=16)
    axs[0].set_xlabel('Epochs', fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot for Train RMSE
    axs[1].plot(epochs, metrics['train_rmse'], label='Train RMSE', color='orange')
    axs[1].plot(epochs, metrics['val_rmse'], label='Validation RMSE', color='red')
    axs[1].set_title(f'Trial {trial.number} - Train RMSE', fontsize=16)
    axs[1].set_xlabel('Epochs', fontsize=14)
    axs[1].set_ylabel('RMSE', fontsize=14)
    axs[1].legend()
    axs[1].grid(True)

    # Subplot for Validation RMSE
    axs[2].plot(epochs, metrics['val_rmse'], label='Validation RMSE', color='purple')
    axs[2].plot(epochs, metrics['val_mae'], label='Validation MAE', color='brown')
    axs[2].set_title(f'Trial {trial.number} - Validation RMSE', fontsize=16)
    axs[2].set_xlabel('Epochs', fontsize=14)
    axs[2].set_ylabel('RMSE', fontsize=14)
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    save_and_close_plot(os.path.join(output_dir, f'best_trial_{trial.number}_metrics.png'), fig)

def print_final_epoch_results(all_trial_metrics, output_dir):
    print("Final Epoch Metrics for Each Trial:")
    for trial_number, metrics in all_trial_metrics:
        last_epoch_index = len(metrics['train_loss']) - 1
        print(f"Results for Trial {trial_number}:")
        print(f"  Training Loss: {metrics['train_loss'][last_epoch_index]:.6f}")
        print(f"  Training RMSE: {metrics['train_rmse'][last_epoch_index]:.6f}")
        print(f"  Training MAE: {metrics['train_mae'][last_epoch_index]:.6f}")
        print(f"  Validation Loss: {metrics['val_loss'][last_epoch_index]:.6f}")
        print(f"  Validation RMSE: {metrics['val_rmse'][last_epoch_index]:.6f}")
        print(f"  Validation MAE: {metrics['val_mae'][last_epoch_index]:.6f}")
        print("-" * 40)

def plot_map_with_test_patches(cropped_raster_data, bbox_coordinates, transform, output_dir):
    # Extract the last band of cropped_raster_data as the target map
    target_map_data = cropped_raster_data[-1]

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Calculate the extent of the original raster for plotting
    height, width = target_map_data.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)

    # Plot the target map
    ax.imshow(target_map_data, cmap='gray', extent=(left, right, bottom, top), interpolation='none')

    # Function to create a semi-transparent rectangle for each patch
    def create_patch_rectangle(bbox, color, label=None):
        # Calculate the width and height of the patch
        width = bbox[1][0] - bbox[0][0]
        height = bbox[0][1] - bbox[2][1]
        # Create and add the rectangle to the plot
        rect = mpatches.Rectangle((bbox[0][0], bbox[2][1]), width, height, linewidth=1, edgecolor=color, facecolor=color, alpha=0.3, label=label)
        ax.add_patch(rect)

    # Overlay the patches with semi-transparent rectangles
    for bbox in bbox_coordinates:
        create_patch_rectangle(bbox, 'red', label='Test Patch' if bbox == bbox_coordinates[0] else "")

    # Avoid repeating labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Test Area with Overlaid Patches')
    save_and_close_plot(os.path.join(output_dir, 'test_patches.png'), fig)
    return fig, ax

def plot_training_validation_loss(trial, output_dir):
    # Extracting metrics from the trial
    metrics = trial.user_attrs['epoch_metrics']
    epochs = range(1, len(metrics['train_loss']) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for Loss
    ax.plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')
    ax.plot(epochs, metrics['val_loss'], label='Validation Loss', color='green')
    ax.set_title('Training and Validation Loss', fontsize=16, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontname='Times New Roman', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # Setting font properties for ticks
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')

    plt.tight_layout()
    save_and_close_plot(os.path.join(output_dir, 'training_validation_loss.png'), fig)


def plot_evaluation_metrics(filtered_alos, filtered_srtm, filtered_nasadem, filtered_aster, filtered_target, filtered_improved, 
                            trendline_alos_target, trendline_srtm_target, trendline_nasadem_target, trendline_aster_target, trendline_improved_target, 
                            rmse_alos_target, mae_alos_target, r2_alos_target, mbe_alos_target, 
                            rmse_srtm_target, mae_srtm_target, r2_srtm_target, mbe_srtm_target, 
                            rmse_nasadem_target, mae_nasadem_target, r2_nasadem_target, mbe_nasadem_target, 
                            rmse_aster_target, mae_aster_target, r2_aster_target, mbe_aster_target, 
                            rmse_improved_target, mae_improved_target, r2_improved_target, mbe_improved_target, output_dir):

    # Set the font properties for Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

    # Scatter Plot 1: ALOS 30m vs Target DEM
    axs[0, 0].scatter(filtered_alos, filtered_target, alpha=0.5)
    axs[0, 0].plot(filtered_alos, trendline_alos_target, color='red')
    axs[0, 0].set_title('ALOS 30m vs Target DEM')
    axs[0, 0].set_xlabel('ALOS 30m')
    axs[0, 0].set_ylabel('Target DEM')
    axs[0, 0].text(0.55, 0.95, f'R2: {r2_alos_target:.2f}\nRMSE: {rmse_alos_target:.2f} m\nMAE: {mae_alos_target:.2f} m\nMBE: {mbe_alos_target:.2f} m', 
                transform=axs[0, 0].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Scatter Plot 2: SRTM vs Target DEM
    axs[0, 1].scatter(filtered_srtm, filtered_target, alpha=0.5)
    axs[0, 1].plot(filtered_srtm, trendline_srtm_target, color='red')
    axs[0, 1].set_title('SRTM 30m vs Target DEM')
    axs[0, 1].set_xlabel('SRTM 30m')
    axs[0, 1].set_ylabel('Target DEM')
    axs[0, 1].text(0.55, 0.95, f'R2: {r2_srtm_target:.2f}\nRMSE: {rmse_srtm_target:.2f} m\nMAE: {mae_srtm_target:.2f} m\nMBE: {mbe_srtm_target:.2f} m', 
                transform=axs[0, 1].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Scatter Plot 3: NASADEM vs Target DEM
    axs[0, 2].scatter(filtered_nasadem, filtered_target, alpha=0.5)
    axs[0, 2].plot(filtered_nasadem, trendline_nasadem_target, color='red')
    axs[0, 2].set_title('NASADEM 30m vs Target DEM')
    axs[0, 2].set_xlabel('NASADEM 30m')
    axs[0, 2].set_ylabel('Target DEM')
    axs[0, 2].text(0.55, 0.95, f'R2: {r2_nasadem_target:.2f}\nRMSE: {rmse_nasadem_target:.2f} m\nMAE: {mae_nasadem_target:.2f} m\nMBE: {mbe_nasadem_target:.2f} m', 
                transform=axs[0, 2].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Scatter Plot 4: ASTER vs Target DEM
    axs[1, 0].scatter(filtered_aster, filtered_target, alpha=0.5)
    axs[1, 0].plot(filtered_aster, trendline_aster_target, color='red')
    axs[1, 0].set_title('ASTER 30m vs Target DEM')
    axs[1, 0].set_xlabel('ASTER 30m')
    axs[1, 0].set_ylabel('Target DEM')
    axs[1, 0].text(0.55, 0.95, f'R2: {r2_aster_target:.2f}\nRMSE: {rmse_aster_target:.2f} m\nMAE: {mae_aster_target:.2f} m\nMBE: {mbe_aster_target:.2f} m', 
                transform=axs[1, 0].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Scatter Plot 5: Improved DEM vs Target DEM
    axs[1, 1].scatter(filtered_improved, filtered_target, alpha=0.5, color='green')
    axs[1, 1].plot(filtered_improved, trendline_improved_target, color='red')
    axs[1, 1].set_title('Improved DEM vs Target DEM')
    axs[1, 1].set_xlabel('Improved DEM')
    axs[1, 1].set_ylabel('Target DEM')
    axs[1, 1].text(0.05, 0.95, f'R2: {r2_improved_target:.2f}\nRMSE: {rmse_improved_target:.2f} m\nMAE: {mae_improved_target:.2f} m\nMBE: {mbe_improved_target:.2f} m', 
                transform=axs[1, 1].transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    save_and_close_plot(os.path.join(output_dir, 'scatter_plots.png'), fig)
