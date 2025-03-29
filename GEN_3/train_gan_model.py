# Training script with modifications for colorful variations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np

# Load models from previous script
from sketch_to_image_gan import Generator, Discriminator
from train_sketch_gan import SketchToImageDataset, transform

# Clear unused GPU memory
torch.cuda.empty_cache()

# Hyperparameters
num_epochs = 100  # Increased epochs, adjust as needed for testing
batch_size = 4
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 64  # Dimension for noise vector

# Initialize models
generator = Generator(noise_dim=noise_dim).to(device)
discriminator = Discriminator(input_channels=4).to(device)  # 1 channel sketch + 3 channel image

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
style_loss_weight = 100.0  # Weight for style loss

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Load dataset
dataset = SketchToImageDataset("C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_3\\dataset\\sketches", 
                             "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_3\\dataset\\real_images", 
                             transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Color diversity loss - encourages varied colors
def color_diversity_loss(images):
    # Calculate the standard deviation of color values across the image
    # Higher standard deviation means more varied colors
    return -torch.mean(torch.std(images, dim=[2, 3]))

# Training loop
for epoch in range(num_epochs):
    for i, (sketches, real_images) in enumerate(data_loader):
        sketches, real_images = sketches.to(device), real_images.to(device)
        batch_size = sketches.size(0)
        
        # Create random noise
        noise = torch.randn(batch_size, noise_dim, sketches.size(2), sketches.size(3), device=device)
        
        # -----------------
        # Train Discriminator
        # -----------------
        d_optimizer.zero_grad()
        
        # Train on real images
        real_outputs = discriminator(sketches, real_images)
        # Create real labels with the same size as discriminator output
        real_labels = torch.ones_like(real_outputs) * 0.9  # 0.9 instead of 1 for label smoothing
        d_real_loss = adversarial_loss(real_outputs, real_labels)
        
        # Train on fake images
        fake_images = generator(sketches, noise)
        fake_outputs = discriminator(sketches, fake_images.detach())
        # Create fake labels with the same size as discriminator output
        fake_labels = torch.zeros_like(fake_outputs) + 0.1  # 0.1 instead of 0 for label smoothing
        d_fake_loss = adversarial_loss(fake_outputs, fake_labels)
        
        # Combined discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        # Train Generator
        # -----------------
        g_optimizer.zero_grad()
        
        # Generate fake images again
        fake_images = generator(sketches, noise)
        fake_outputs = discriminator(sketches, fake_images)
        
        # Adversarial loss - use real_labels with correct shape
        g_adv_loss = adversarial_loss(fake_outputs, real_labels)
        
        # Content loss - reduced weight to allow more variation
        g_content_loss = l1_loss(fake_images, real_images) * 10.0
        
        # Color diversity loss
        g_diversity_loss = color_diversity_loss(fake_images) * 0.5
        
        # Total generator loss
        g_loss = g_adv_loss + g_content_loss + g_diversity_loss
        
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if i % 20 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                  f"G_adv: {g_adv_loss.item():.4f}, G_content: {g_content_loss.item():.4f}, "
                  f"G_diversity: {g_diversity_loss.item():.4f}")
    
    # Save generated images
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Generate images with different noise vectors
    with torch.no_grad():
        fixed_sketches = sketches[:4].clone()  # Take first 4 sketches
        variations = []
        
        # Generate 3 variations of each sketch
        for j in range(3):  
            noise = torch.randn(fixed_sketches.size(0), noise_dim, 
                               fixed_sketches.size(2), fixed_sketches.size(3), device=device)
            fake = generator(fixed_sketches, noise)
            variations.append(fake)
        
        # Convert sketches from 1 channel to 3 channels by repeating the grayscale values
        fixed_sketches_rgb = fixed_sketches.repeat(1, 3, 1, 1)
        
        # Save grid with original sketches and variations
        grid_images = torch.cat([fixed_sketches_rgb] + variations, dim=0)
        save_image(grid_images, f"outputs/epoch_{epoch}_variations.png", nrow=4, normalize=True)
    
    # Save model checkpoints
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")
    
    # Save latest models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

print("Training complete!")