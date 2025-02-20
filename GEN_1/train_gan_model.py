import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Load models from previous script
from sketch_to_image_gan import Generator, Discriminator
from train_sketch_gan import SketchToImageDataset, transform

# Clear unused GPU memory
torch.cuda.empty_cache()

# Hyperparameters
num_epochs = 30
batch_size = 4  # Reduced batch size to prevent OOM errors
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Load dataset
dataset = SketchToImageDataset("dataset/sketches", "dataset/real_images", transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (sketches, real_images) in enumerate(data_loader):
        sketches, real_images = sketches.to(device), real_images.to(device)

        # Train Discriminator
        real_preds = discriminator(real_images)
        fake_images = generator(sketches)
        fake_preds = discriminator(fake_images.detach())

        real_labels = torch.ones_like(real_preds).to(device)
        fake_labels = torch.zeros_like(fake_preds).to(device)

        real_loss = adversarial_loss(real_preds, real_labels)
        fake_loss = adversarial_loss(fake_preds, fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        fake_preds = discriminator(fake_images)
        g_adv_loss = adversarial_loss(fake_preds, real_labels)
        g_l1_loss = l1_loss(fake_images, real_images) * 100  # L1 loss weight
        g_loss = g_adv_loss + g_l1_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Save generated images
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    save_image(fake_images, f"outputs/epoch_{epoch}.png", normalize=True)

    # Save model checkpoints
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

print("Training complete!")
