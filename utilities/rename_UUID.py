import os
import uuid
import argparse

# Set RECURSIVE as a global variable (you can still modify this manually if needed)
RECURSIVE = True  # Set to False if you only want to rename top-level files

def rename_files_with_uuid(directory, recursive=False):
    script_name = os.path.basename(__file__)

    # Ensure the path exists
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory")
        return

    # Walk through directories if recursive mode is enabled
    for root, _, files in (os.walk(directory) if recursive else [(directory, None, os.listdir(directory))]):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip directories and the script itself
            if not os.path.isfile(file_path) or filename == script_name:
                continue

            # Split filename and extension
            _, extension = os.path.splitext(filename)

            # Generate new unique filename
            while True:
                new_name = f"{uuid.uuid4().hex}{extension}"
                new_path = os.path.join(root, new_name)
                if not os.path.exists(new_path):
                    break

            # Rename the file
            try:
                os.rename(file_path, new_path)
                print(f"Renamed '{file_path}' to '{new_path}'")
            except Exception as e:
                print(f"Error renaming '{file_path}': {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Rename files in a directory with UUIDs.")
    parser.add_argument(
        "--rename_folder",
        type=str,
        required=True,
        help="The directory containing files to rename"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the rename function with the provided directory
    rename_files_with_uuid(args.rename_folder, RECURSIVE)

if __name__ == "__main__":
    main()