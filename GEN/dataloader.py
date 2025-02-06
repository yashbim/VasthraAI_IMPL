import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Custom Dataset to load paired images
class SketchDataset(Dataset):
    def __init__(self, sketch_dir, real_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.real_dir = real_dir
        self.transform = transform
        self.filenames = os.listdir(sketch_dir)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.filenames[idx])
        real_path = os.path.join(self.real_dir, self.filenames[idx])
        
        sketch = Image.open(sketch_path).convert("L")  # Grayscale
        real = Image.open(real_path).convert("RGB")
        
        if self.transform:
            sketch = self.transform(sketch)
            real = self.transform(real)
        
        return sketch, real

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between [-1, 1]
])

# Create DataLoader
sketch_dir = "C:\\Users\\Bimsara\\Documents\\fyp\\IPD\\VasthraAI_POC\\GEN\\Dataset\\sketches"
real_dir = "C:\\Users\\Bimsara\\Documents\\fyp\\IPD\\VasthraAI_POC\\GEN\\Dataset\\real_images"
dataset = SketchDataset(sketch_dir, real_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"Loaded {len(dataset)} image pairs")
