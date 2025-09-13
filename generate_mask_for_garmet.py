from transformers import AutoModelForImageSegmentation
import torch
import os
from PIL import Image
from torchvision import transforms

def load_birefnet():
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()
    birefnet.half()
    return birefnet

def remove_background(image, moddelib):
    # Data settings
    image = image.copy()
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()
    # Prediction
    with torch.no_grad():
        preds = moddelib['birefnet'](input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image

def add_padding_to_bbox(bbox, padding, image_size):
    """Add padding to bounding box while keeping it within image bounds"""
    left, top, right, bottom = bbox
    width, height = image_size
    
    # Add padding
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    
    return (left, top, right, bottom)

def process_folder(foldername, moddelib, padding=5):
    """Process a single folder"""
    folder_path = f"simple_data2/{foldername}"
    garment_path = os.path.join(folder_path, "garment.jpg")
    
    # Check if garment.jpg exists
    if not os.path.exists(garment_path):
        print(f"Warning: {garment_path} not found, skipping folder {foldername}")
        return
    
    try:
        # Load and process the image
        garment_image = Image.open(garment_path)
        
        # Remove background and get mask
        garment_with_alpha = remove_background(garment_image, moddelib)
        garment_mask = garment_with_alpha.split()[3]  # Get alpha channel
        
        # Get bounding box and add padding
        bbox = garment_mask.getbbox()
        if bbox is None:
            print(f"Warning: No object detected in {garment_path}, skipping")
            return
            
        padded_bbox = add_padding_to_bbox(bbox, padding, garment_image.size)
        
        # Crop with padding
        clipped_garment_image = garment_image.crop(padded_bbox)
        # clipped_garment_with_alpha = garment_with_alpha.crop(padded_bbox)
        
        # Save images
        mask_save_path = os.path.join(folder_path, "garment_mask.png")
        clipped_save_path = os.path.join(folder_path, "clipped_garment.png")
        
        garment_mask.save(mask_save_path)
        clipped_garment_image.save(clipped_save_path)
        
        print(f"Processed {foldername}: saved mask and clipped garment with {padding}px padding")
        
    except Exception as e:
        print(f"Error processing {foldername}: {str(e)}")

def process_all_folders(padding=5):
    """Process all folders in simple_data directory"""
    
    # Load the model once
    print("Loading BiRefNet model...")
    moddelib = {'birefnet': load_birefnet()}
    print("Model loaded successfully!")
    
    # Get all folders in simple_data
    simple_data_path = "simple_data2"
    if not os.path.exists(simple_data_path):
        print(f"Error: {simple_data_path} directory not found")
        return
    
    folders = [f for f in os.listdir(simple_data_path) 
               if os.path.isdir(os.path.join(simple_data_path, f))]
    
    if not folders:
        print("No folders found in simple_data directory")
        return
    
    print(f"Found {len(folders)} folders to process")
    
    # Process each folder
    for i, foldername in enumerate(folders, 1):
        print(f"Processing folder {i}/{len(folders)}: {foldername}")
        process_folder(foldername, moddelib, padding)
    
    print("Batch processing completed!")

# Run the batch processing
if __name__ == "__main__":
    # You can adjust the padding value here (default is 5px)
    process_all_folders(padding=5)