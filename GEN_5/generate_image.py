import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import argparse
import numpy as np
from sketch_to_image_gan import Generator  # Import your enhanced Generator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator
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

def preprocess_sketch(sketch_path):
    """Apply preprocessing to enhance sketch quality before generation."""
    sketch = Image.open(sketch_path).convert("L")  # Load as grayscale
    
    # Convert to numpy for processing
    sketch_np = np.array(sketch)
    
    # Enhance contrast
    sketch_np = np.clip((sketch_np.astype(np.float32) - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    enhanced_sketch = Image.fromarray(sketch_np)
    
    return enhanced_sketch

def generate_image(sketch_path, output_dir="generated_images", enhance_sketch=True, ensemble=True):
    """Generate an image from a sketch with optional enhancements."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output file path with the timestamp
    output_filename = f"generated_image_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save original sketch for comparison
    sketch_output_path = os.path.join(output_dir, f"input_sketch_{timestamp}.png")
    
    # Preprocess the sketch if enhancement is enabled
    if enhance_sketch:
        sketch = preprocess_sketch(sketch_path)
        sketch.save(sketch_output_path)
    else:
        sketch = Image.open(sketch_path).convert("L")
        sketch.save(sketch_output_path)
    
    # Transform the sketch
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    # Generate with ensemble if enabled (average multiple generations with small noise)
    if ensemble:
        with torch.no_grad():
            # Generate multiple versions with noise and average them
            num_samples = 3
            generated_images = []
            
            for _ in range(num_samples):
                # Add small noise to sketch input
                noise = torch.randn_like(sketch_tensor) * 0.02
                noisy_sketch = sketch_tensor + noise
                
                # Generate image
                gen_img = generator(noisy_sketch)
                generated_images.append(gen_img)
            
            # Average the generated images
            generated_image = torch.mean(torch.stack(generated_images), dim=0)
    else:
        # Single generation
        with torch.no_grad():
            generated_image = generator(sketch_tensor)

    # Post-process: Convert output tensor to image
    generated_image = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2  # Convert to [0,1]
    generated_image = (generated_image * 255).astype("uint8")  # Convert to uint8

    # Save the image
    Image.fromarray(generated_image).save(output_path)
    print(f"Generated image saved at: {output_path}")
    print(f"Input sketch saved at: {sketch_output_path}")
    
    return output_path, sketch_output_path

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a sketch.")
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
        "--enhance",
        action='store_true',
        help="Apply sketch enhancement preprocessing."
    )
    parser.add_argument(
        "--ensemble",
        action='store_true',
        help="Use ensemble generation for better results."
    )
    args = parser.parse_args()

    # Generate the image
    generate_image(args.sketch_path, args.output_dir, args.enhance, args.ensemble)