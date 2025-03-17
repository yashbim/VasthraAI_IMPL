import cv2
import os

def generate_sketch(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges
    
    cv2.imwrite(output_path, edges)  # Save the sketch

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            generate_sketch(input_path, output_path)
            print(f"Processed: {filename}")

# Example usage
input_folder = "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\real_images"  # Change to your actual folder
output_folder = "C:\\Users\\Bimsara\\Documents\\fyp\\VasthraAI_IMPL\\GEN_2\\dataset\\sketches"
process_folder(input_folder, output_folder)
