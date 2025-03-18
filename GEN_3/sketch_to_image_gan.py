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
    def __init__(self, noise_dim=8):  # Extra noise input for variation
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        # Encoder (extract features from sketch)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Fully connected layer to mix noise with feature map
        self.fc_noise = nn.Linear(noise_dim, 64 * 128 * 128)  # Adjust for image size

        # Residual blocks for refinement
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])

        # Decoder (upsample to generate image)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, sketch, noise):
        x = self.encoder(sketch)

        # Reshape and inject noise
        noise = self.fc_noise(noise).view(-1, 64, 128, 128)  # Reshape to match feature map
        x = x + noise  # Inject noise to introduce variations

        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
