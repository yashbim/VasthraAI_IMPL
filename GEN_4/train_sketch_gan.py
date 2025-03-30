
#DATALOADER

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Define dataset class
class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, real_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.real_dir = real_dir
        self.transform = transform
        self.image_filenames = os.listdir(sketch_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.image_filenames[idx])
        real_path = os.path.join(self.real_dir, self.image_filenames[idx])
        
        sketch = Image.open(sketch_path).convert("L")  # Grayscale
        real = Image.open(real_path).convert("RGB")  # Color
        
        if self.transform:
            sketch = self.transform(sketch)
            real = self.transform(real)
        
        return sketch, real

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load dataset
dataset = SketchToImageDataset("C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\sketches", "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\real_images", transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Check batch
for sketch, real in data_loader:
    print(f"Sketch shape: {sketch.shape}, Real shape: {real.shape}")
    break
