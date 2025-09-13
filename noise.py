import os
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import Image
from tqdm import tqdm

def setup_pipeline():
    """Initialize the image-to-image pipeline"""
    print("Loading pipeline...")
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    
    # Create image-to-image pipeline from the text-to-image pipeline
    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")
    print("Pipeline loaded successfully!")
    return pipeline

def process_image_multiple_strengths(pipeline, image_path, output_dir, base_filename, strengths, prompt="", guidance_scale=10.5):
    """Process a single image with multiple strength values"""
    try:
        # Load the input image once
        init_image = load_image(image_path)
        og_size = init_image.size
        resized_image = init_image.resize((1024, 1024))
        
        success_count = 0
        
        # Process with each strength value
        for strength in strengths:
            # Generate output filename with strength value
            strength_str = str(strength).replace('.', '_')  # 0.1 -> 0_1
            output_filename = f"{base_filename}_str{strength_str}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # Generate the new image
                result = pipeline(
                    prompt=prompt, 
                    image=resized_image, 
                    strength=strength, 
                    guidance_scale=guidance_scale,
                    num_inference_steps=30
                ).images[0]
                
                # Resize back to original size and save
                result = result.resize(og_size)
                result.save(output_path)
                success_count += 1
                
            except Exception as e:
                tqdm.write(f"✗ Error processing {image_path} with strength {strength}: {str(e)}")
        
        return success_count
        
    except Exception as e:
        tqdm.write(f"✗ Error loading {image_path}: {str(e)}")
        return 0

def process_folder(root_folder, strengths, prompt="", guidance_scale=10.5):
    """Process all subfolders in the root directory"""
    
    # Setup pipeline
    pipeline = setup_pipeline()
    
    # First pass: count total images to process
    total_images = 0
    subdirs_to_process = []
    
    for subdir, dirs, files in os.walk(root_folder):
        # Skip the root directory itself
        if subdir == root_folder:
            continue
            
        images_in_subdir = []
        garment_path = os.path.join(subdir, "garment.jpg")
        model_path = os.path.join(subdir, "model.jpg")
        
        if os.path.exists(garment_path):
            images_in_subdir.append(("garment.jpg", garment_path, "garment_img2img"))
            total_images += 1
        
        if os.path.exists(model_path):
            images_in_subdir.append(("model.jpg", model_path, "model_img2img"))
            total_images += 1
            
        if images_in_subdir:
            subdirs_to_process.append((subdir, images_in_subdir))
    
    if total_images == 0:
        print("No images found to process!")
        return
    
    total_generations = total_images * len(strengths)
    print(f"Found {total_images} images to process in {len(subdirs_to_process)} folders")
    print(f"Will generate {total_generations} total images ({len(strengths)} strengths per image)")
    print(f"Strength values: {strengths}")
    
    # Track statistics
    total_processed = 0
    total_errors = 0
    
    # Create main progress bar
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        # Process each subfolder
        for subdir, images_to_process in subdirs_to_process:
            folder_name = os.path.basename(subdir)
            pbar.set_description(f"Processing folder: {folder_name}")
            
            # Process each image in the subfolder
            for img_name, img_path, base_output_name in images_to_process:
                
                # Update progress bar description
                pbar.set_description(f"Processing {folder_name}/{img_name}")
                
                success_count = process_image_multiple_strengths(
                    pipeline, img_path, subdir, base_output_name, strengths, prompt, guidance_scale
                )
                
                total_processed += success_count
                total_errors += (len(strengths) - success_count)
                
                # Update progress bar
                pbar.update(1)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total images generated: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Success rate: {total_processed/(total_processed + total_errors)*100:.1f}%" if (total_processed + total_errors) > 0 else "N/A")
    print(f"{'='*50}")

def main():
    """Main function to run the batch processing"""
    
    # Configuration
    ROOT_FOLDER = "vton"  # Change this to your folder path
    PROMPT = ""  # Your prompt here (empty string for no prompt)
    STRENGTHS = [0.1, 0.2, 0.3]  # List of strength values to process
    GUIDANCE_SCALE = 10.5  # How closely to follow the prompt
    
    # Validate root folder exists
    if not os.path.exists(ROOT_FOLDER):
        print(f"Error: Root folder '{ROOT_FOLDER}' does not exist!")
        print("Please update the ROOT_FOLDER variable with the correct path.")
        return
    
    print(f"Starting batch processing...")
    print(f"Root folder: {ROOT_FOLDER}")
    print(f"Prompt: '{PROMPT}' (empty = no prompt)")
    print(f"Strengths: {STRENGTHS}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"{'='*50}")
    
    # Start processing
    process_folder(ROOT_FOLDER, STRENGTHS, PROMPT, GUIDANCE_SCALE)

if __name__ == "__main__":
    main()