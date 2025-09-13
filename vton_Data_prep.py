# import os
# import json
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PIL import Image, ImageDraw
# import requests
# from io import BytesIO

# # Load dataset
# dataset = load_dataset("datahiveai/Virtual-Try-On-Dataset")

# # Load model
# model = AutoModelForCausalLM.from_pretrained(
#     "vikhyatk/moondream2",
#     revision="2025-06-21",
#     trust_remote_code=True,
#     device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
# )

# def url_to_pil_image(url):
#     """Convert URL to PIL Image"""
#     if url == None:
#         return None
#     try:
#         # Download the image
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()  # Raises an HTTPError for bad responses
        
#         # Convert to PIL Image
#         image = Image.open(BytesIO(response.content))
        
#         return image
        
#     except requests.RequestException as e:
#         raise requests.RequestException(f"Error downloading image from {url}: {e}")
#     except Exception as e:
#         raise Exception(f"Error processing image: {e}")

# def draw_bbox_with_padding(image, bbox_list, padding=0.05, fill_color='black'):
#     """
#     Draw bounding boxes on an image with padding and fill them with a color.
    
#     Args:
#         image (PIL.Image): The input image
#         bbox_list (list): List of bounding box dictionaries with normalized coordinates
#         padding (float): Padding to add around each bbox (as fraction of image size)
#         fill_color (str): Color to fill the bounding boxes
        
#     Returns:
#         PIL.Image: Image with bounding boxes drawn
#     """
#     # Create a copy of the image to avoid modifying the original
#     img_copy = image.copy()
#     draw = ImageDraw.Draw(img_copy)
    
#     # Get image dimensions
#     img_width, img_height = image.size
    
#     for bbox in bbox_list:
#         # Convert normalized coordinates to pixel coordinates
#         x_min_px = int(bbox['x_min'] * img_width)
#         y_min_px = int(bbox['y_min'] * img_height)
#         x_max_px = int(bbox['x_max'] * img_width)
#         y_max_px = int(bbox['y_max'] * img_height)
        
#         # Add padding
#         pad_x = int(padding * img_width)
#         pad_y = int(padding * img_height)
        
#         x_min_padded = max(0, x_min_px - pad_x)
#         y_min_padded = max(0, y_min_px - pad_y)
#         x_max_padded = min(img_width, x_max_px + pad_x)
#         y_max_padded = min(img_height, y_max_px + pad_y)
        
#         # Draw filled rectangle
#         draw.rectangle(
#             [(x_min_padded, y_min_padded), (x_max_padded, y_max_padded)],
#             fill=fill_color
#         )
    
#     return img_copy

# def create_directory_if_needed(save_dir):
#     """Create directory only when needed"""
#     os.makedirs(save_dir, exist_ok=True)
#     return save_dir

# def save_sample_data(save_dir, model_img, garment_img, bboxed_img, type2, metadata):
#     """Save all sample data to the specified directory"""
#     # Create directory only when we're actually saving
#     create_directory_if_needed(save_dir)
    
#     # Save images
#     if model_img:
#         model_img.save(os.path.join(save_dir, "model.jpg"))
#     if garment_img:
#         garment_img.save(os.path.join(save_dir, "garment.jpg"))
#     if bboxed_img:
#         bboxed_img.save(os.path.join(save_dir, "bboxed_model.jpg"))
    
#     # Save metadata
#     sample_info = {
#         "type2": type2,
#         "metadata": metadata
#     }
    
#     with open(os.path.join(save_dir, "info.json"), 'w') as f:
#         json.dump(sample_info, f, indent=2)

# def process_sample(index, base_save_dir="vton"):
#     """Process a single sample from the dataset"""
#     # Get sample data
#     sample = dataset['train'][index]
#     id_val = sample['model-id']
#     category = sample['type1']
#     model_front_url = sample['model-front']
#     model_back_url = sample['model-back']
#     garment_front_url = sample['garment-front']
#     garment_back_url = sample['garment-back']
#     type1 = sample['type1']
#     type2 = sample['type2']
    
#     # Create metadata
#     metadata = {
#         "index": index,
#         "model_id": id_val,
#         "type1": type1,
#         "type2": type2,
#         "category": category,
#         "urls": {
#             "model_front": model_front_url,
#             "model_back": model_back_url,
#             "garment_front": garment_front_url,
#             "garment_back": garment_back_url
#         }
#     }
    
#     # Define directory paths (but don't create them yet)
#     front_dir = os.path.join(base_save_dir, f"{id_val}_front")
#     back_dir = os.path.join(base_save_dir, f"{id_val}_back")
    
#     # Load images
#     model_front = url_to_pil_image(model_front_url)
#     model_back = url_to_pil_image(model_back_url)
#     garment_front = url_to_pil_image(garment_front_url)
#     garment_back = url_to_pil_image(garment_back_url)
    
#     results = {}
    
#     # Process front images
#     if model_front and garment_front:
#         try:
#             objects = model.detect(model_front, type2)["objects"]
#             if len(objects) > 0:
#                 bboxed_front = draw_bbox_with_padding(model_front, objects, padding=0.0001)
                
#                 # Save front data
#                 save_sample_data(
#                     front_dir, 
#                     model_front, 
#                     garment_front, 
#                     bboxed_front, 
#                     type2, 
#                     metadata
#                 )
                
#                 results['front'] = {
#                     'processed': True,
#                     'save_dir': front_dir,
#                     'objects_detected': len(objects)
#                 }
                
#                 print(f"✓ Front images saved to: {front_dir}")
#             else:
#                 print(f"⚠ No objects detected in front image for sample {index}")
#                 results['front'] = {'processed': False, 'reason': 'No objects detected'}
#         except Exception as e:
#             print(f"✗ Error processing front images for sample {index}: {e}")
#             results['front'] = {'processed': False, 'reason': str(e)}
#     else:
#         print(f"⚠ Missing front images for sample {index}")
#         results['front'] = {'processed': False, 'reason': 'Missing images'}
    
#     # Process back images
#     if model_back and garment_back:
#         try:
#             objects = model.detect(model_back, type2)["objects"]
#             if len(objects) > 0:
#                 bboxed_back = draw_bbox_with_padding(model_back, objects, padding=0.0001)
                
#                 # Save back data
#                 save_sample_data(
#                     back_dir, 
#                     model_back, 
#                     garment_back, 
#                     bboxed_back, 
#                     type2, 
#                     metadata
#                 )
                
#                 results['back'] = {
#                     'processed': True,
#                     'save_dir': back_dir,
#                     'objects_detected': len(objects)
#                 }
                
#                 print(f"✓ Back images saved to: {back_dir}")
#             else:
#                 print(f"⚠ No objects detected in back image for sample {index}")
#                 results['back'] = {'processed': False, 'reason': 'No objects detected'}
#         except Exception as e:
#             print(f"✗ Error processing back images for sample {index}: {e}")
#             results['back'] = {'processed': False, 'reason': str(e)}
#     else:
#         print(f"⚠ Missing back images for sample {index}")
#         results['back'] = {'processed': False, 'reason': 'Missing images'}
    
#     return results

# def process_multiple_samples(start_index=0, num_samples=10, base_save_dir="vton"):
#     """Process multiple samples from the dataset"""
#     print(f"Processing {num_samples} samples starting from index {start_index}")
#     print(f"Saving to base directory: {base_save_dir}")
#     print("=" * 50)
    
#     all_results = {}
    
#     for i in range(start_index, start_index + num_samples):
#         if i >= len(dataset['train']):
#             print(f"Reached end of dataset at index {i}")
#             break
            
#         print(f"\nProcessing sample {i}...")
#         try:
#             results = process_sample(i, base_save_dir)
#             all_results[i] = results
#         except Exception as e:
#             print(f"✗ Failed to process sample {i}: {e}")
#             all_results[i] = {'error': str(e)}
    
#     # Save processing summary only if we have results
#     if all_results:
#         os.makedirs(base_save_dir, exist_ok=True)  # Create base dir only when saving summary
#         summary_path = os.path.join(base_save_dir, "processing_summary.json")
#         with open(summary_path, 'w') as f:
#             json.dump(all_results, f, indent=2)
#         print(f"Processing complete! Summary saved to: {summary_path}")
#     else:
#         print("No samples were processed successfully.")
    
#     return all_results

# # Main execution
# if __name__ == "__main__":
#     # Create base directory
#     os.makedirs("vton", exist_ok=True)
    
#     # Process single sample (your original example)
#     # print("Processing single sample (index 1):")
#     # single_result = process_sample(1)
    
#     # Uncomment below to process multiple samples
#     print("\n" + "="*50)
#     print(f"Processing multiple samples:{len(dataset['train'])}")
#     multiple_results = process_multiple_samples(start_index=0, num_samples=len(dataset['train']))



import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def load_model():
    """Load the moondream2 model for captioning"""
    print("Loading moondream2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        device_map={"": "cuda"}  # Change to 'mps' for Apple Silicon or 'cpu' for CPU
    )
    print("Model loaded successfully!")
    return model

def generate_captions_for_directory(model, directory_path):
    """Generate captions for all images in a directory"""
    captions = {}
    
    # Define the image files to process
    image_files = {
        'model': 'model.jpg',
    }
    
    print(f"Processing directory: {directory_path}")
    
    for image_type, filename in image_files.items():
        image_path = os.path.join(directory_path, filename)
        
        if os.path.exists(image_path):
            try:
                # Load image
                image = Image.open(image_path)
                
                # Generate caption
                print(f"  Generating caption for {filename}...")
                
                caption = model.caption(image, length="normal")['caption']
                
                captions[image_type] = {
                    'filename': filename,
                    'caption': caption
                }
                
                print(f"    ✓ Caption generated: {caption[:50]}...")
                
            except Exception as e:
                print(f"    ✗ Error processing {filename}: {e}")
                captions[image_type] = {
                    'filename': filename,
                    'caption': None,
                    'error': str(e)
                }
        else:
            print(f"    ⚠ {filename} not found")
            captions[image_type] = {
                'filename': filename,
                'caption': None,
                'error': 'File not found'
            }
    
    return captions

def save_captions(directory_path, captions):
    """Save captions to a JSON file in the directory"""
    captions_path = os.path.join(directory_path, "captions.json")
    
    try:
        with open(captions_path, 'w') as f:
            json.dump(captions, f, indent=2)
        print(f"  ✓ Captions saved to: captions.json")
        return True
    except Exception as e:
        print(f"  ✗ Error saving captions: {e}")
        return False

def update_info_json(directory_path, captions):
    """Update the existing info.json file to include caption information"""
    info_path = os.path.join(directory_path, "info.json")
    
    if os.path.exists(info_path):
        try:
            # Load existing info
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            # Add captions section
            info['captions'] = captions
            info['captions_generated'] = True
            
            # Save updated info
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"  ✓ Updated info.json with caption information")
            return True
            
        except Exception as e:
            print(f"  ✗ Error updating info.json: {e}")
            return False
    else:
        print(f"  ⚠ info.json not found in {directory_path}")
        return False

def find_processed_directories(base_dir="vton"):
    """Find all processed directories in the base directory"""
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found!")
        return []
    
    directories = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a processed directory (should have info.json)
            info_path = os.path.join(item_path, "info.json")
            if os.path.exists(info_path):
                directories.append(item_path)
    
    return directories

def check_existing_captions(directory_path):
    """Check if captions already exist for this directory"""
    captions_path = os.path.join(directory_path, "captions.json")
    info_path = os.path.join(directory_path, "info.json")
    return False, "no captions found"
    # Check for captions.json
    if os.path.exists(captions_path):
        return True, "captions.json exists"
    
    # Check if info.json has captions
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            if 'captions' in info:
                return True, "captions in info.json"
        except:
            pass
    
    return False, "no captions found"

def process_all_directories(base_dir="vton", skip_existing=True):
    """Process all directories and add captions"""
    print("="*60)
    print("ADDING CAPTIONS TO EXISTING DATASET")
    print("="*60)
    
    # Find all processed directories
    directories = find_processed_directories(base_dir)
    
    if not directories:
        print(f"No processed directories found in '{base_dir}'")
        return
    
    print(f"Found {len(directories)} processed directories")
    
    # Load model
    model = load_model()
    
    # Process statistics
    stats = {
        'total': len(directories),
        'processed': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Process each directory
    for i, directory in enumerate(directories, 1):
        directory_name = os.path.basename(directory)
        print(f"\n[{i}/{len(directories)}] Processing: {directory_name}")
        
        # Check if captions already exist
        has_captions, caption_status = check_existing_captions(directory)
        
        if has_captions and skip_existing:
            print(f"  ⏭ Skipping (already has captions: {caption_status})")
            stats['skipped'] += 1
            continue
        
        try:
            # Generate captions
            captions = generate_captions_for_directory(model, directory)
            
            # Save captions
            caption_success = save_captions(directory, captions)
            # info_success = update_info_json(directory, captions)
            
            if caption_success:
                stats['processed'] += 1
                print(f"  ✓ Successfully processed {directory_name}")
            else:
                stats['errors'] += 1
                print(f"  ⚠ Partial success for {directory_name}")
                
        except Exception as e:
            stats['errors'] += 1
            print(f"  ✗ Error processing {directory_name}: {e}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("CAPTION GENERATION COMPLETE!")
    print("="*60)
    print(f"Total directories: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (already had captions): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    # Save processing log
    log_path = os.path.join(base_dir, "caption_processing_log.json")
    try:
        with open(log_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Processing log saved to: {log_path}")
    except Exception as e:
        print(f"Could not save processing log: {e}")

def process_specific_directories(base_dir="vton", directory_names=None):
    """Process specific directories by name"""
    if directory_names is None:
        print("No directory names specified")
        return
    
    print(f"Processing specific directories: {directory_names}")
    
    # Load model
    model = load_model()
    
    for dir_name in directory_names:
        directory_path = os.path.join(base_dir, dir_name)
        
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            continue
        
        print(f"\nProcessing: {dir_name}")
        
        try:
            # Generate captions
            captions = generate_captions_for_directory(model, directory_path)
            
            # Save captions
            save_captions(directory_path, captions)
            update_info_json(directory_path, captions)
            
            print(f"✓ Successfully processed {dir_name}")
            
        except Exception as e:
            print(f"✗ Error processing {dir_name}: {e}")

# Main execution
if __name__ == "__main__":
    # Option 1: Process all directories (default)
    # This will find all processed directories and add captions to them
    process_all_directories(base_dir="vton", skip_existing=True)
    
    # Option 2: Process specific directories
    # Uncomment the lines below if you want to process only specific directories
    # specific_dirs = ["sample_1_front", "sample_1_back"]  # Replace with your directory names
    # process_specific_directories(base_dir="vton", directory_names=specific_dirs)
    
    # Option 3: Force reprocess all (including those with existing captions)
    # Uncomment the line below to regenerate captions even for directories that already have them
    # process_all_directories(base_dir="vton", skip_existing=False)