from PIL import Image
import os

def process_images(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(input_path) as img:
                    # Resize the image to 512x512
                    new_size = (512, 512)
                    resized_img = img.resize(new_size, Image.LANCZOS)

                    # Convert to RGB if necessary
                    if resized_img.mode != 'RGB':
                        resized_img = resized_img.convert('RGB')

                    # Save the processed image in the output directory
                    output_path = os.path.join(output_dir, filename)
                    resized_img.save(output_path)

                    print(f"Processed {filename} and saved to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Define input and output directories
input_directory = 'C:\\Users\\Bimsara\\Documents\\fyp\\IPD\\VasthraAI_POC\\GEN\\Dataset\\raw_images'
output_directory = 'C:\\Users\\Bimsara\\Documents\\fyp\\IPD\\VasthraAI_POC\\GEN\\Dataset\\real_images'

# Run the function
process_images(input_directory, output_directory)

print("All images processed successfully!")
