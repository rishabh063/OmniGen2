import os
import shutil
from pathlib import Path

def copy_files_between_folders(folder_a, folder_b, file_names):
    """
    Copy files with specified names from subfolders in folder_a 
    to matching subfolders in folder_b
    
    Args:
        folder_a (str): Source folder path
        folder_b (str): Destination folder path  
        file_names (list): List of file names to copy (e.g., ['config.txt', 'data.json'])
    """
    
    folder_a_path = Path(folder_a)
    folder_b_path = Path(folder_b)
    
    # Check if both folders exist
    if not folder_a_path.exists():
        print(f"Error: Folder A '{folder_a}' does not exist")
        return
    if not folder_b_path.exists():
        print(f"Error: Folder B '{folder_b}' does not exist")
        return
    
    copied_count = 0
    
    # Iterate through all subfolders in folder A
    for subfolder_a in folder_a_path.iterdir():
        if subfolder_a.is_dir():
            subfolder_name = subfolder_a.name
            subfolder_b = folder_b_path / subfolder_name
            
            # Check if matching subfolder exists in folder B
            if subfolder_b.exists() and subfolder_b.is_dir():
                print(f"Processing folder: {subfolder_name}")
                
                # Find and copy matching files by name
                for file_path in subfolder_a.iterdir():
                    if file_path.is_file() and file_path.name in file_names:
                        destination = subfolder_b / file_path.name
                        
                        try:
                            shutil.copy2(file_path, destination)
                            print(f"  Copied: {file_path.name}")
                            copied_count += 1
                        except Exception as e:
                            print(f"  Error copying {file_path.name}: {e}")
            else:
                print(f"Warning: No matching folder '{subfolder_name}' found in folder B")
    
    print(f"\nCompleted! Copied {copied_count} files total.")

# Example usage
if __name__ == "__main__":
    # Set your folder paths here
    FOLDER_A = "simple_data2"
    FOLDER_B = "Simple_data3"
    
    # Specify which file types to copy (change as needed)
    file_names = ['clipped_garment.png', 'garment_mask.png']  
    
    copy_files_between_folders(FOLDER_A, FOLDER_B, file_names)
