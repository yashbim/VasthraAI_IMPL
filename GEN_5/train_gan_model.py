import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np

# Load models from previous script
from sketch_to_image_gan import Generator, MultiScaleDiscriminator
from train_sketch_gan import SketchToImageDataset, transform

# Clear unused GPU memory
torch.cuda.empty_cache()

# Hyperparameters
num_epochs = 100  # Increased for better convergence
batch_size = 4
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_pixel = 100  # L1 loss weight
lambda_feat = 10    # Feature matching loss weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = MultiScaleDiscriminator().to(device)

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Feature matching loss
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))
    return loss / len(real_features)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate schedulers
g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=num_epochs, eta_min=lr*0.1)
d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=num_epochs, eta_min=lr*0.1)

# Load dataset
dataset = SketchToImageDataset("C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\sketches", 
                             "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\real_images", 
                             transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create output directories
if not os.path.exists("outputs"):
    os.makedirs("outputs")
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Training loop
for epoch in range(num_epochs):
    for i, (sketches, real_images) in enumerate(data_loader):
        sketches, real_images = sketches.to(device), real_images.to(device)
        batch_size = sketches.size(0)

        # Generate images
        fake_images = generator(sketches)

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()
        
        # Compute losses for real images at multiple scales
        real_preds_1, real_preds_2 = discriminator(real_images)
        real_labels = torch.ones_like(real_preds_1).to(device)
        
        # Compute losses for fake images at multiple scales
        fake_preds_1, fake_preds_2 = discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(fake_preds_1).to(device)
        
        # Combine losses from different scales
        d_real_loss = (adversarial_loss(real_preds_1, real_labels) + 
                     adversarial_loss(real_preds_2, real_labels)) / 2
        d_fake_loss = (adversarial_loss(fake_preds_1, fake_labels) + 
                     adversarial_loss(fake_preds_2, fake_labels)) / 2
        
        # Add some noise to labels for more stable training
        real_labels = torch.FloatTensor(batch_size, 1, 16, 16).uniform_(0.7, 1.0).to(device)
        fake_labels = torch.FloatTensor(batch_size, 1, 16, 16).uniform_(0.0, 0.3).to(device)
        
        d_loss = d_real_loss + d_fake_loss
        
        d_loss.backward()
        d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        g_optimizer.zero_grad()
        
        # Compute adversarial loss
        fake_preds_1, fake_preds_2 = discriminator(fake_images)
        g_adv_loss = (adversarial_loss(fake_preds_1, real_labels) + 
                    adversarial_loss(fake_preds_2, real_labels)) / 2
        
        # Pixel-wise loss
        g_pixel_loss = l1_loss(fake_images, real_images) * lambda_pixel
        
        # Total generator loss
        g_loss = g_adv_loss + g_pixel_loss
        
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if i % 20 == 0:  # Increased frequency for better monitoring
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], "
                 f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                 f"G Pixel Loss: {g_pixel_loss.item()/lambda_pixel:.4f}")
    
    # Update learning rates
    g_scheduler.step()
    d_scheduler.step()
    
    # Save generated images periodically
    if (epoch+1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            # Generate a batch of images and save them
            fake_images = generator(sketches)
            img_grid = torch.cat((sketches.repeat(1, 3, 1, 1), fake_images, real_images), dim=0)
            save_image(img_grid, f"outputs/epoch_{epoch+1}.png", nrow=batch_size, normalize=True)
    
    # Save model checkpoints periodically
    if (epoch+1) % 10 == 0 or (epoch+1) == num_epochs:
        torch.save({
            'epoch': epoch+1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
        
        # Always save the latest model
        torch.save(generator.state_dict(), "generator.pth")
        torch.save(discriminator.state_dict(), "discriminator.pth")

print("Training complete!")