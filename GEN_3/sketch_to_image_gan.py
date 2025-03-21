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
    def __init__(self, input_channels=1, output_channels=3, num_residual_blocks=9, noise_dim=64):
        super(Generator, self).__init__()
        # Increased number of residual blocks for more transformation power
        
        # First layer processes sketch
        self.sketch_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Noise processing branch
        self.noise_processor = nn.Sequential(
            nn.Conv2d(noise_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for transformation
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )
        
        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x, noise=None):
        # Generate noise if none provided
        batch_size, _, height, width = x.size()
        if noise is None:
            noise = torch.randn(batch_size, 64, height, width, device=x.device)
        
        # Process sketch input
        x = self.sketch_encoder(x)
        
        # Process and add noise
        noise = self.noise_processor(noise)
        x = x + noise
        
        # Downsample
        x = self.down1(x)
        x = self.down2(x)
        
        # Apply residual blocks
        x = self.res_blocks(x)
        
        # Upsample
        x = self.up1(x)
        x = self.up2(x)
        
        # Generate final output
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=4):  # Changed to 4 to accept sketch+real/fake image
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input is concatenated sketch and generated/real image
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, sketch, img):
        # Concatenate sketch and image along channel dimension
        x = torch.cat([sketch, img], dim=1)
        return self.model(x)