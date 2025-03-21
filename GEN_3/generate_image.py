import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import argparse
import numpy as np
from sketch_to_image_gan import Generator  # Import your trained Generator

# Load the trained generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(noise_dim=64).to(device)  # Updated to include noise_dim
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel input
    transforms.Resize((512, 512)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

def generate_image(sketch_path, output_dir="generated_images", num_variations=3, seed=None):
    """Generate multiple variations of an image from a sketch and save them."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate timestamp base for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and preprocess the sketch
    sketch = Image.open(sketch_path).convert("L")  # Load as grayscale
    sketch = transform(sketch).unsqueeze(0).to(device)  # Add batch dimension

    # Generate multiple variations of the image
    generated_images = []
    for i in range(num_variations):
        # Create random noise
        noise = torch.randn(1, 64, sketch.size(2), sketch.size(3), device=device)
        
        # Generate the image
        with torch.no_grad():
            generated_image = generator(sketch, noise)
        
        # Convert output tensor to image
        img_array = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2  # Convert to [0,1]
        img_array = (img_array * 255).astype("uint8")  # Convert to uint8
        
        # Create image from array
        img = Image.fromarray(img_array)
        
        # Save the image
        output_filename = f"generated_image_{timestamp}_var{i+1}.png"
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)
        print(f"Generated variation {i+1} saved at: {output_path}")
        
        generated_images.append(img)
    
    # Also save a grid view of all variations
    grid_width = min(4, num_variations)
    grid_height = (num_variations + grid_width - 1) // grid_width
    cell_size = 512
    grid_img = Image.new('RGB', (grid_width * cell_size, grid_height * cell_size))
    
    for i, img in enumerate(generated_images):
        grid_x = (i % grid_width) * cell_size
        grid_y = (i // grid_width) * cell_size
        grid_img.paste(img, (grid_x, grid_y))
    
    grid_path = os.path.join(output_dir, f"grid_variations_{timestamp}.png")
    grid_img.save(grid_path)
    print(f"Grid of all variations saved at: {grid_path}")
    
    return generated_images

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple variations of an image from a sketch.")
    parser.add_argument(
        "--sketch_path",
        type=str,
        required=True,
        help="Path to the input sketch image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Directory to save the generated image. Default is 'generated_images'."
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="Number of variations to generate. Default is 3."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default is None (random)."
    )
    args = parser.parse_args()

    # Generate the images
    generate_image(args.sketch_path, args.output_dir, args.variations, args.seed)