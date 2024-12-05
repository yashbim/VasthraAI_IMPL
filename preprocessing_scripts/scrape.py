import os
import shutil

def copy_images(source_dir, target_dir):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file is a .png or .jpg (case insensitive)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct full file paths
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # Copy the file to the target directory
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

    print("All PNG and JPG images have been copied successfully.")

# Example usage
source_directory = "/home/lenovo/fyp/VasthraAI_POC/batiks"  # Replace with your source directory path
target_directory = "/home/lenovo/fyp/VasthraAI_POC/preprocessed_dataset"  # Replace with your target directory path

copy_images(source_directory, target_directory)
