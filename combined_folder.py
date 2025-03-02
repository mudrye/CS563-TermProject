import os
import shutil

def copy_files_to_combined_folder(source_parent_folder, combined_folder):
    """Copy all images and JSON files from source folders to a combined folder."""
    # Ensure the combined folder exists
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Get all folder names from the parent directory
    source_folders = [f for f in os.listdir(source_parent_folder) if os.path.isdir(os.path.join(source_parent_folder, f))]
    
    # Loop through each source folder
    for source_folder in source_folders:
        source_folder_path = os.path.join(source_parent_folder, source_folder)
        if os.path.exists(source_folder_path):
            # Loop through files in the source folder
            for file_name in os.listdir(source_folder_path):
                # Check if the file is an image (.jpg) or JSON (.json)
                if file_name.endswith('.jpg') or file_name.endswith('.json'):
                    source_file = os.path.join(source_folder_path, file_name)
                    destination_file = os.path.join(combined_folder, file_name)
                    # Copy the file to the combined folder
                    shutil.copy(source_file, destination_file)
                    print(f"Copied {file_name} to {combined_folder}")
        else:
            print(f"Warning: Source folder {source_folder} does not exist.")

# Example usage
source_parent_folder = 'sample_rico/rico_sampled_dataset'  # Replace with the path to the parent directory containing the folders
combined_folder = 'sample_rico/combined_folder'  # Folder where you want to save the combined files

copy_files_to_combined_folder(source_parent_folder, combined_folder)
