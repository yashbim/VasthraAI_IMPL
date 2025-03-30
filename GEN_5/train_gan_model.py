import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import numpy as np
from train_sketch_gan import SketchToImageDataset, transform
from sketch_to_image_gan import Generator, MultiScaleDiscriminator

# Clear unused GPU memory
torch.cuda.empty_cache()

# Hyperparameters
num_epochs = 100  # Increased for better convergence
batch_size = 4
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_pixel = 100  # L1 loss weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = MultiScaleDiscriminator().to(device)

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate schedulers
g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=num_epochs, eta_min=lr*0.1)
d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=num_epochs, eta_min=lr*0.1)

# Load dataset
dataset = SketchToImageDataset("C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_5\\dataset\\sketches", 
                             "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_5\\dataset\\real_images", 
                             transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create output directories
if not os.path.exists("outputs"):
    os.makedirs("outputs")
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Print first batch shapes for debugging
for sketches, real_images in data_loader:
    print(f"Sketch shape: {sketches.shape}, Real shape: {real_images.shape}")
    break

# Training loop
for epoch in range(num_epochs):
    for i, (sketches, real_images) in enumerate(data_loader):
        sketches, real_images = sketches.to(device), real_images.to(device)
        batch_size = sketches.size(0)

        # Generate fake images
        fake_images = generator(sketches)

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()
        
        # Get discriminator outputs for real and fake images
        real_preds_1, real_preds_2 = discriminator(real_images)
        fake_preds_1, fake_preds_2 = discriminator(fake_images.detach())
        
        # Create labels for loss calculation
        # Make sure we use the correct shape based on the discriminator's output
        real_labels_1 = torch.ones_like(real_preds_1).to(device)
        real_labels_2 = torch.ones_like(real_preds_2).to(device)
        fake_labels_1 = torch.zeros_like(fake_preds_1).to(device)
        fake_labels_2 = torch.zeros_like(fake_preds_2).to(device)
        
        # Add label smoothing for more stable training
        real_labels_1 = real_labels_1 * 0.9
        real_labels_2 = real_labels_2 * 0.9
        
        # Calculate discriminator losses
        d_real_loss = (adversarial_loss(real_preds_1, real_labels_1) + 
                      adversarial_loss(real_preds_2, real_labels_2)) / 2
        d_fake_loss = (adversarial_loss(fake_preds_1, fake_labels_1) + 
                      adversarial_loss(fake_preds_2, fake_labels_2)) / 2
        
        d_loss = d_real_loss + d_fake_loss
        
        d_loss.backward()
        d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        g_optimizer.zero_grad()
        
        # Get discriminator outputs for generated images
        fake_preds_1, fake_preds_2 = discriminator(fake_images)
        
        # Adversarial loss wants the generator to fool the discriminator
        g_adv_loss = (adversarial_loss(fake_preds_1, real_labels_1) + 
                     adversarial_loss(fake_preds_2, real_labels_2)) / 2
        
        # Pixel-wise loss between generated and real images
        g_pixel_loss = l1_loss(fake_images, real_images) * lambda_pixel
        
        # Total generator loss
        g_loss = g_adv_loss + g_pixel_loss
        
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if i % 20 == 0:
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
            # Repeat single-channel sketches to 3 channels for visualization
            sketches_3c = sketches.repeat(1, 3, 1, 1) 
            img_grid = torch.cat((sketches_3c, fake_images, real_images), dim=0)
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