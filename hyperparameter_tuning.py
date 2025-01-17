# hyperparameter_tuning.py

import torch
import os
import random
import numpy as np
import optuna
import torch.optim as optim
from tqdm import tqdm
from cnn_unet import UNet, MSELoss, calculate_rmse, calculate_mae
from data_preparation import prepared_and_split_data
from visualization import visualize_trial_metrics, get_params_for_trial, visualize_best_trial_metrics, print_final_epoch_results, plot_training_validation_loss
from config import PATCH_SIZE, OVERLAP_PERCENTAGE, BATCH_SIZE, BEST_MODEL_PATH, NUM_EPOCHS, NO_DATA_VALUE, BASE_OUTPUT_DIR, OUTPUT_DIRECTORY_PLOTS


# Ensure the base output directory exists
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# Ensure the plots directory exists
if not os.path.exists(OUTPUT_DIRECTORY_PLOTS):
    os.makedirs(OUTPUT_DIRECTORY_PLOTS)

# Ensure the directory for the best model exists
best_model_dir = os.path.dirname(BEST_MODEL_PATH)
if not os.path.exists(best_model_dir):
    os.makedirs(best_model_dir)

# Path to save the best model
best_model_path = BEST_MODEL_PATH

# Global variable to track the best loss
best_loss = float('inf')
all_trial_metrics = []

# Objective function
def objective(trial, raster_data, original_transform, test_area_gdf):
    global best_loss
    # Set random seed for reproducibility within each trial
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter suggestions by Optuna for U-Net
    num_epochs = NUM_EPOCHS
    depth = 4
    base_filters = 64
    kernel_size = 3
    use_dropout_down = False
    use_dropout_bottleneck = True
    use_dropout_up = False

    # Conditionally add dropout_rate to the hyperparameters
    if use_dropout_down or use_dropout_bottleneck or use_dropout_up:
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    else:
        dropout_rate = 0.0

    # Batch size and learning rate suggestions
    batch_size = BATCH_SIZE
    learning_rate = trial.suggest_float('learning_rate', 0.00015, 0.00025, log=True)
    patch_size = PATCH_SIZE
    overlap_percentage = OVERLAP_PERCENTAGE

    # Load and prepare data
    train_loader, validation_loader, test_loader, features, features_patches, target_patches, num_patches_x, num_patches_y, bbox_coordinates, cropped_raster_data, train_indices, val_indices, test_patches_idx, new_transform = prepared_and_split_data(
        patch_size, raster_data, overlap_percentage, batch_size, original_transform, test_area_gdf, random_state=42)

    # Initialize the U-Net model with the suggested hyperparameters
    unet_model = UNet(
        in_channels=39,
        out_channels=1,
        depth=depth,
        base_filters=base_filters,
        kernel_size=kernel_size,
        pool_size=2,
        dropout_rate=dropout_rate,
        use_dropout_down=use_dropout_down,
        use_dropout_bottleneck=use_dropout_bottleneck,
        use_dropout_up=use_dropout_up
    ).to(device)

    # Define the loss function and optimizer
    criterion = MSELoss()
    optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate)

    # Metrics storage
    epoch_metrics = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': []
    }

    # Training and validation loop
    for epoch in tqdm(range(num_epochs), desc=f'Trial {trial.number} - Epoch Progress', leave=False):
        # Training loop
        unet_model.train()
        total_train_loss, total_train_rmse, total_train_mae = 0.0, 0.0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = unet_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_rmse += calculate_rmse(outputs.detach(), targets.detach())
            total_train_mae += calculate_mae(outputs.detach(), targets.detach())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_rmse = total_train_rmse / len(train_loader)
        avg_train_mae = total_train_mae / len(train_loader)

        # Validation loop
        unet_model.eval()
        total_val_loss, total_val_rmse, total_val_mae = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = unet_model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                total_val_rmse += calculate_rmse(outputs.detach(), targets.detach())
                total_val_mae += calculate_mae(outputs.detach(), targets.detach())

        avg_val_loss = total_val_loss / len(validation_loader)
        avg_val_rmse = total_val_rmse / len(validation_loader)
        avg_val_mae = total_val_mae / len(validation_loader)

        # Save the model if it's the best one so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(unet_model.state_dict(), best_model_path)

        # Store metrics for this epoch
        epoch_metrics['train_loss'].append(avg_train_loss)
        epoch_metrics['train_rmse'].append(avg_train_rmse)
        epoch_metrics['train_mae'].append(avg_train_mae)
        epoch_metrics['val_loss'].append(avg_val_loss)
        epoch_metrics['val_rmse'].append(avg_val_rmse)
        epoch_metrics['val_mae'].append(avg_val_mae)
        trial.set_user_attr("epoch_metrics", epoch_metrics)

    # Append the metrics and trial number to the list
    all_trial_metrics.append((trial.number, epoch_metrics))

    # Return the primary metric for Optuna to optimize
    return avg_val_loss

# Function to run hyperparameter tuning
def run_hyperparameter_tuning(raster_data, original_transform, test_area_gdf, n_trials=1):
    global best_loss
    best_loss = float('inf')
    study = optuna.create_study(direction='minimize')
    with tqdm(total=n_trials, desc='Hyperparameter Tuning Progress') as trial_bar:
        for _ in range(n_trials):
            study.optimize(lambda trial: objective(trial, raster_data, original_transform, test_area_gdf), n_trials=1)
            trial_bar.update(1)


    # After the optimization is completed
    for trial_number, metrics in all_trial_metrics:
        params = get_params_for_trial(study, trial_number)
        visualize_trial_metrics(trial_number, metrics, params, OUTPUT_DIRECTORY_PLOTS)

    print_final_epoch_results(all_trial_metrics, OUTPUT_DIRECTORY_PLOTS)
    best_trial = study.best_trial
    visualize_best_trial_metrics(best_trial, best_trial.params, OUTPUT_DIRECTORY_PLOTS)

    # Plot and save the training and validation loss for the best trial
    plot_training_validation_loss(best_trial, OUTPUT_DIRECTORY_PLOTS)

    return study
