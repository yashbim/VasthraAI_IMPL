import torch
from torchvision import transforms
from PIL import Image
import os
from sketch_to_image_gan import Generator  # Import your trained Generator

# Load the trained generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel input
    transforms.Resize((512, 512)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

def generate_image(sketch_path, output_path="generated_image.png"):
    """Generate an image from a sketch."""
    sketch = Image.open(sketch_path).convert("L")  # Load as grayscale
    sketch = transform(sketch).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        generated_image = generator(sketch)
    
    # Convert output tensor to image
    generated_image = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2  # Convert to [0,1]
    generated_image = (generated_image * 255).astype("uint8")  # Convert to uint8
    
    # Save the image
    Image.fromarray(generated_image).save(output_path)
    print(f"Generated image saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    sketch_file = "path/to/your/sketch.png"  # Replace with your test sketch
    generate_image(sketch_file)
