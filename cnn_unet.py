# unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_dropout, dropout_rate):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode='replicate')
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode='replicate')
        self.activation2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters, depth, kernel_size, pool_size, dropout_rate, use_dropout_down, use_dropout_bottleneck, use_dropout_up):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * (2 ** (i - 1))
            out_ch = base_filters * (2 ** i)
            self.down_blocks.append(ConvBlock(in_ch, out_ch, kernel_size, use_dropout_down, dropout_rate))
        self.pool = nn.MaxPool2d(pool_size)

        # Bottleneck
        bottleneck_channels = base_filters * (2 ** (depth - 1))
        self.bottleneck = ConvBlock(bottleneck_channels, bottleneck_channels * 2, kernel_size, use_dropout_bottleneck, dropout_rate)

        # Expanding Path (Decoder)
        self.up_transpose_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_channels * 2 if i == depth - 1 else base_filters * (2 ** i) * 2
            out_ch = base_filters * (2 ** i)
            self.up_transpose_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_conv_blocks.append(ConvBlock(out_ch * 2, out_ch, kernel_size, use_dropout_up, dropout_rate))

        # Final Convolution
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Expanding Path
        skip_connections = reversed(skip_connections)
        for up_transpose, up_conv, skip_connection in zip(self.up_transpose_blocks, self.up_conv_blocks, skip_connections):
            x = up_transpose(x)
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1)
            x = up_conv(x)

        # Final Convolution
        x = self.final(x)
        return x

# Loss Functions
class MSELoss(nn.Module):
    def __init__(self, no_data_val=-9999):
        super().__init__()
        self.no_data_val = no_data_val
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        mask = target != self.no_data_val
        loss = self.mse(output, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()

class MAELoss(nn.Module):
    def __init__(self, no_data_val=-9999):
        super().__init__()
        self.no_data_val = no_data_val
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        mask = target != self.no_data_val
        loss = self.mae(output, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()

class RMSELoss(nn.Module):
    def __init__(self, no_data_val=-9999):
        super().__init__()
        self.no_data_val = no_data_val
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        mask = target != self.no_data_val
        loss = self.mse(output, target)
        masked_loss = loss * mask
        mean_mse = masked_loss.sum() / mask.sum()
        rmse = torch.sqrt(mean_mse)
        return rmse

# Metrics calculation functions
def calculate_rmse(outputs, targets, no_data_val=-9999):
    mask = targets != no_data_val
    mse_loss = nn.MSELoss(reduction='none')
    mse = mse_loss(outputs, targets)
    masked_mse = mse * mask
    mean_mse = masked_mse.sum() / mask.sum()
    rmse = torch.sqrt(mean_mse)
    return rmse.item()

def calculate_mae(outputs, targets, no_data_val=-9999):
    mask = targets != no_data_val
    mae_loss = nn.L1Loss(reduction='none')
    mae = mae_loss(outputs, targets)
    masked_mae = mae * mask
    return (masked_mae.sum() / mask.sum()).item()

def calculate_mse(outputs, targets, no_data_val=-9999):
    mask = targets != no_data_val
    mse_loss = nn.MSELoss(reduction='none')
    mse = mse_loss(outputs, targets)
    masked_mse = mse * mask
    if mask.sum() > 0:
        return (masked_mse.sum() / mask.sum()).item()
    else:
        return float('nan')