import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import argparse
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

def generate_image(sketch_path, output_dir="generated_images"):
    """Generate an image from a sketch and save it with a timestamp in a specific folder."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output file path with the timestamp
    output_filename = f"generated_image_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Load and preprocess the sketch
    sketch = Image.open(sketch_path).convert("L")  # Load as grayscale
    sketch = transform(sketch).unsqueeze(0).to(device)  # Add batch dimension

    # Generate the image
    with torch.no_grad():
        generated_image = generator(sketch)

    # Convert output tensor to image
    generated_image = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2  # Convert to [0,1]
    generated_image = (generated_image * 255).astype("uint8")  # Convert to uint8

    # Save the image
    Image.fromarray(generated_image).save(output_path)
    print(f"Generated image saved at: {output_path}")

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
    args = parser.parse_args()

    # Generate the image
    generate_image(args.sketch_path, args.output_dir)