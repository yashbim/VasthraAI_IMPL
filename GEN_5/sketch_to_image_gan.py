import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_filters=64, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_filters*4) for _ in range(num_residual_blocks)])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
        # Attention module
        self.attention = SpatialAttention(base_filters*4)
    
    def forward(self, x):
        # Initial and downsampling
        x = self.initial(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        
        # Apply attention
        x2 = self.attention(x2)
        
        # Residual blocks
        x2 = self.res_blocks(x2)
        
        # Upsampling with skip connections
        x = self.up1(x2)
        x = self.up2(x)
        
        # Output
        x = self.output(x)
        
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        
    def forward(self, x):
        # Create attention map
        attn = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))
        
        # Apply attention
        return x * attn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(Discriminator, self).__init__()
        
        # Initial layer without batch norm
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Middle layers
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(base_filters*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        
        # Two discriminators at different scales
        self.disc_original = Discriminator(input_channels)
        
        # Downsample before passing to the second discriminator
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.disc_downscaled = Discriminator(input_channels)
        
    def forward(self, x):
        # Get results at original scale
        d1 = self.disc_original(x)
        
        # Get results at downscaled resolution
        x_downscaled = self.downsample(x)
        d2 = self.disc_downscaled(x_downscaled)
        
        return d1, d2