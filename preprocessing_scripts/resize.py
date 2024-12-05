from PIL import Image
import os

def resize_images_in_directory(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(input_path) as img:
                    # Get original dimensions
                    width, height = img.size

                    # Only resize if both dimensions are >= 256
                    if width >= 256 and height >= 256:
                        new_size = (512, 512)
                        resized_img = img.resize(new_size, Image.LANCZOS)
                        
                        # Save the resized image in the output directory
                        output_path = os.path.join(output_dir, filename)
                        resized_img.save(output_path)
                        print(f"Resized {filename} to {new_size} and saved to {output_path}")
                    else:
                        print(f"Skipped {filename} (smaller than 512x512)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
input_directory = '/home/lenovo/fyp/VasthraAI_POC/preprocessed_dataset'
output_directory = '/home/lenovo/fyp/VasthraAI_POC/resized_images'
resize_images_in_directory(input_directory, output_directory)
