import os
import torch
import facer
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize device and models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

def get_bounding_box(mask):
    coords = np.argwhere(mask)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return [x0, y0, x1, y1]
    return None

def rescale_faces(faces, scale):
    faces_rescaled = {
        'rects': faces['rects'].clone(),
        'points': faces['points'].clone(),
        'scores': faces['scores'].clone(),
        'image_ids': faces['image_ids'].clone()
    }
    faces_rescaled['rects'] *= scale
    faces_rescaled['points'] *= scale
    return faces_rescaled

def limit_image_size_complex(img, max_size=1000000):
    ratio = (max_size / (img.width * img.height))**0.5
    if ratio >= 1:
        return img, 1
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    img = img.resize((new_width, new_height))
    return img, ratio

def process_single_image(image_path, output_path):
    """Process a single image and save the result"""
    try:
        # Load image
        ogimage = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device == 'cuda' else torch.no_grad():
                # Resize image if too large
                image_resized, scale_factor = limit_image_size_complex(ogimage)
                np_image = torch.from_numpy(np.array(image_resized))
                image_tensor = facer.hwc2bchw(np_image).to(device=device)
                
                # Detect faces
                with torch.inference_mode():
                    faces = face_detector(image_tensor)
                
                # Rescale faces back to original size
                faces = rescale_faces(faces, 1/scale_factor)
                
                # Parse faces on original image
                np_image = torch.from_numpy(np.array(ogimage))
                image_tensor = facer.hwc2bchw(np_image).to(device=device)
                
                with torch.inference_mode():
                    faces = face_parser(image_tensor, faces)
        
        # Generate segmentation mask
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
        vis_img = vis_seg_probs.sum(0, keepdim=True)
        
        # Convert to PIL and save
        tensor_to_pil = T.ToPILImage()
        current_mask = tensor_to_pil(vis_img)
        current_mask.save(output_path)
        
        logger.info(f"Successfully processed: {image_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return False

def process_folder_structure(root_folder, output_root=None):
    """
    Process all images in the folder structure
    
    Args:
        root_folder (str): Path to the root folder containing subfolders with images
        output_root (str): Optional output root folder. If None, saves results next to original images
    """
    root_path = Path(root_folder)
    
    if not root_path.exists():
        logger.error(f"Root folder does not exist: {root_folder}")
        return
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    processed_count = 0
    failed_count = 0
    
    # Walk through all subfolders
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        logger.info(f"Processing folder: {subfolder.name}")
        
        # Create output folder if specified
        if output_root:
            output_subfolder = Path(output_root) / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)
        else:
            output_subfolder = subfolder
        
        # Process all images in the subfolder
        for image_file in subfolder.iterdir():
            if image_file.suffix.lower() in image_extensions:
                # Generate output filename
                output_filename = f"{image_file.stem}_result.png"
                output_path = output_subfolder / output_filename
                
                # Process the image
                if process_single_image(str(image_file), str(output_path)):
                    processed_count += 1
                else:
                    failed_count += 1
    
    logger.info(f"Processing complete! Processed: {processed_count}, Failed: {failed_count}")

def main():
    # Configuration
    ROOT_FOLDER = "human_Dataset/Original Images"  # Change this to your root folder path
    OUTPUT_FOLDER = None  # Set to a path if you want results in a separate folder structure
    
    # Alternative: Save results in a separate folder structure
    # OUTPUT_FOLDER = "human_Dataset/Processed_Results"
    
    logger.info("Starting batch face parsing processing...")
    process_folder_structure(ROOT_FOLDER, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()