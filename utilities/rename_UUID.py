import os
import uuid
import argparse

def rename_files_with_uuid(directory):
    script_name = os.path.basename(__file__)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories and the script itself
        if not os.path.isfile(file_path) or filename == script_name:
            continue
        
        # Split filename and extension
        _, extension = os.path.splitext(filename)
        
        # Generate new unique filename
        while True:
            new_name = f"{uuid.uuid4().hex}{extension}"
            new_path = os.path.join(directory, new_name)
            if not os.path.exists(new_path):
                break
        
        # Rename the file
        os.rename(file_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files in a directory with random UUIDs.')
    parser.add_argument('directory', help='Path to the directory containing files to rename')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        exit(1)
    
    rename_files_with_uuid(args.directory)