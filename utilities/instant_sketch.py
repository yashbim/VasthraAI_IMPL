import cv2
import os
import sys

def generate_sketch(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges
    
    cv2.imwrite(output_path, edges)  # Save the sketch
    print(f"Sketch generated and saved to: {output_path}")

def process_single_image(image_path):
    # Get the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # Get the original filename and create output filename
    filename = os.path.basename(image_path)
    output_filename = f"sketch_{os.path.splitext(filename)[0]}.png"
    output_path = os.path.join(desktop_path, output_filename)
    
    # Check if input image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    generate_sketch(image_path, output_path)

if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    process_single_image(image_path)