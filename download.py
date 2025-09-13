import pandas as pd
import requests
import os
import shutil
import time
from tqdm import tqdm
import concurrent.futures

# --- Configuration ---
XLSX_FILE_PATH = 'Sample input data sheet Vero Only W Soch (3).xlsx'  # <--- IMPORTANT: CHANGE THIS TO YOUR XLSX FILE NAME
BASE_OUTPUT_FOLDER = 'nykaa_images'
IMAGE_COLUMNS = ['default', 'back', 'bottom', 'front', 'left', 'right', 'top']
FINAL_STYLE_COLUMN = 'final_style'
MAX_DOWNLOAD_RETRIES = 3 # Reduced default retries for faster feedback in parallel, adjust as needed
RETRY_DELAY_SECONDS = 2
MAX_WORKERS = 10  # Number of parallel download threads. Adjust based on your network & CPU.
                    # Too many might overload the server or your connection.

# --- Helper function to sanitize folder names ---
def sanitize_foldername(name):
    name = str(name)
    name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    name = "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()
    return name

# --- Function to download a single image (will be run in a thread) ---
def download_single_image_task(image_url, image_save_path, img_col_name, final_style_raw, style_folder_path):
    """
    Downloads a single image with retries.
    Returns True on success, False on failure.
    """
    # Ensure the specific style folder exists (should be created before calling this, but as a safeguard)
    # This part could be moved outside if all tasks for a style are grouped, but for individual tasks, it's fine.
    os.makedirs(style_folder_path, exist_ok=True)

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            # tqdm.write(f"  [Style: {final_style_raw}, Img: {img_col_name}] Attempt {attempt}/{MAX_DOWNLOAD_RETRIES} to download {os.path.basename(image_save_path)}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(image_url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            with open(image_save_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            # tqdm.write(f"    [Style: {final_style_raw}, Img: {img_col_name}] Successfully downloaded.")
            return True # Success

        except requests.exceptions.RequestException as e:
            # tqdm.write(f"    [Style: {final_style_raw}, Img: {img_col_name}] Attempt {attempt} failed: {e}")
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                tqdm.write(f"  [!] FAILED (All Retries): Style '{final_style_raw}', Image '{img_col_name}', URL: {image_url}, Error: {e}")
        except Exception as e: # Catch other unexpected errors
            tqdm.write(f"  [!] FAILED (Unexpected Error): Style '{final_style_raw}', Image '{img_col_name}', URL: {image_url}, Error: {e}")
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS) # Still retry for unexpected errors within the attempt loop
            else:
                 tqdm.write(f"  [!] FAILED (All Retries - Unexpected): Style '{final_style_raw}', Image '{img_col_name}'.")
            # For truly critical unexpected errors, you might want to break or handle differently
    return False # Failure after all retries

# --- Main orchestrator script ---
def download_images_parallel(excel_path, output_folder_base):
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: The file '{excel_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    if FINAL_STYLE_COLUMN not in df.columns:
        print(f"Error: Column '{FINAL_STYLE_COLUMN}' not found in the Excel file.")
        return

    os.makedirs(output_folder_base, exist_ok=True)
    print(f"Base output folder: '{output_folder_base}'")
    print(f"Using up to {MAX_WORKERS} parallel workers.")

    tasks_to_submit = []
    style_folders_created = set() # To avoid redundant os.makedirs calls for style folders

    print("Preparing download tasks...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scanning Excel Rows"):
        final_style_raw = row[FINAL_STYLE_COLUMN]

        if pd.isna(final_style_raw):
            # tqdm.write(f"Skipping row {index + 2} due to missing '{FINAL_STYLE_COLUMN}'.") # Can be noisy
            continue
        
        final_style = sanitize_foldername(str(final_style_raw))
        if not final_style:
            # tqdm.write(f"Skipping row {index + 2} as '{FINAL_STYLE_COLUMN}' resulted in empty folder name.") # Can be noisy
            continue

        style_folder_path = os.path.join(output_folder_base, final_style)
        
        # Create style folder once if not already done
        # This is still done sequentially here to ensure folders exist before tasks might try to write
        if style_folder_path not in style_folders_created:
            os.makedirs(style_folder_path, exist_ok=True)
            style_folders_created.add(style_folder_path)

        for img_col_name in IMAGE_COLUMNS:
            if img_col_name not in df.columns:
                continue

            image_url = row[img_col_name]
            if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.strip():
                continue
            image_url = image_url.strip()

            try:
                url_path = image_url.split('?')[0]
                file_extension = os.path.splitext(url_path)[1]
                if not file_extension or len(file_extension) > 5 or len(file_extension) < 2: # Basic check
                    file_extension = '.jpg' # Default
                
                image_filename = f"{img_col_name}{file_extension}"
                image_save_path = os.path.join(style_folder_path, image_filename)
                
                # Add task parameters to the list
                tasks_to_submit.append(
                    (image_url, image_save_path, img_col_name, final_style_raw, style_folder_path)
                )
            except Exception as e:
                tqdm.write(f"Error preparing task for URL '{image_url}' in style '{final_style_raw}': {e}")

    if not tasks_to_submit:
        print("No image URLs found to download.")
        return

    print(f"\nFound {len(tasks_to_submit)} images to download. Starting parallel download...")
    
    successful_downloads = 0
    failed_downloads = 0

    # Using ThreadPoolExecutor to download images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks and store the future objects
        future_to_task_info = {
            executor.submit(download_single_image_task, *task_params): task_params 
            for task_params in tasks_to_submit
        }

        # Process completed tasks using tqdm for progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(tasks_to_submit), desc="Downloading Images"):
            task_params = future_to_task_info[future]
            try:
                success = future.result() # Get result from the thread (True or False)
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1 # Failure already logged by download_single_image_task
            except Exception as exc:
                # This catches exceptions not handled within download_single_image_task or if the task itself failed to start
                img_url_for_error = task_params[0]
                style_for_error = task_params[3]
                tqdm.write(f"  [!] CRITICAL ERROR for task (Style: {style_for_error}, URL: {img_url_for_error}): {exc}")
                failed_downloads += 1
    
    print("\n--- Download process finished. ---")
    print(f"Successfully downloaded: {successful_downloads} images.")
    print(f"Failed to download:      {failed_downloads} images.")


if __name__ == "__main__":
    excel_file = XLSX_FILE_PATH
    
    if excel_file == 'your_file_name.xlsx' and not os.path.exists(excel_file) :
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE THE 'XLSX_FILE_PATH' VARIABLE IN THE SCRIPT         !!!")
        print("!!! with the actual name of your Excel file.                      !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not os.path.exists(excel_file):
        print(f"Error: The specified Excel file '{excel_file}' does not exist.")
        print("Please ensure the 'XLSX_FILE_PATH' variable is set correctly.")
    else:
        start_time = time.time()
        download_images_parallel(excel_file, BASE_OUTPUT_FOLDER)
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")