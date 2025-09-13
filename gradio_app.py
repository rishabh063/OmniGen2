# import dotenv
# dotenv.load_dotenv(override=True)
# import accelerate
# import gradio as gr
# import os
# from typing import List, Tuple, Optional
# from PIL import Image, ImageOps
# import torch
# from torchvision.transforms.functional import to_pil_image, to_tensor
# from accelerate import Accelerator
# from diffusers.hooks import apply_group_offloading
# from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
# from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
# import random

# # Initialize accelerator and pipeline
# accelerator = accelerate.Accelerator()
# model_path = "OmniGen2/OmniGen2"

# # Initialize pipeline once
# pipeline = OmniGen2Pipeline.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
#     model_path,
#     subfolder="transformer",
#     torch_dtype=torch.bfloat16,
# )
# pipeline = pipeline.to(accelerator.device, dtype=torch.bfloat16)

# # Load LoRA weights
# pipeline.unload_lora_weights()
# pipeline.load_lora_weights("experiments/vtonMixbig/checkpoint-100000/transformer_lora")

# # Default negative prompt
# DEFAULT_NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

# # Sample bboxed model images for Type 1


# SAMPLE_BBOXED_IMAGES = {
#     "Model 1": "vton/262208_front/bboxed_model.jpg",
#     "Model 2": "vton/276266_front/bboxed_model.jpg",
#     "Model 3": "vton/269114_front/bboxed_model.jpg",
#     "Model 4": "vton/269114_front/bboxed_model.jpg",
#     "Model 5":"vton/262922_front/bboxed_model.jpg",
#     "Model 6":"vton/272369_front/bboxed_model.jpg",
#     "Model 7":"vton/265829_front/bboxed_model.jpg",
#     "Model 8":"vton/272044_front/bboxed_model.jpg",
#     "Model 9":"vton/266161_front/bboxed_model.jpg",
#     "Model 10":"vton/261401_front/bboxed_model.jpg"    
# }

# # Sample clothing images for Type 1
# SAMPLE_CLOTHING_IMAGES = {
#     "Clothing 1": "example_images/bf5e132d9b04e347365118427592e33e39ae5bad.jpg",
#     "Clothing 2": "example_images/f3979db3a2dab4a04259b0bef35ae5719bb52bc8.jpg",
#     "Clothing 3": "example_images/4442fbac4e3080ec20b2f14e353fea267249b0dd.jpg",
# }

# # Sample images for Type 2
# SAMPLE_MANNEQUIN_IMAGES = {
#     "Jeans Sample": "example_images/833435a34190ab6ee7d9b33f7e37d9345fcf9e17.jpg",
# }

# def load_sample_image(image_dict, key):
#     """Load and return a sample image for display."""
#     if key and key in image_dict:
#         try:
#             return Image.open(image_dict[key])
#         except:
#             return None
#     return None

# def preprocess_images(images: List) -> List[Image.Image]:
#     """Preprocess input images."""
#     processed = []
#     for img in images:
#         if img is not None:
#             if isinstance(img, str):
#                 img = Image.open(img)
#             img = ImageOps.exif_transpose(img)
#             img = img.convert('RGB')
#             processed.append(img)
#     return processed

# def resize_image(image, max_size=1024):
#     """Resize image if needed."""
#     original_width, original_height = image.size
    
#     if original_width <= max_size and original_height <= max_size:
#         return image
    
#     if original_width > original_height:
#         new_width = max_size
#         new_height = int((original_height * max_size) / original_width)
#     else:
#         new_height = max_size
#         new_width = int((original_width * max_size) / original_height)
    
#     resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
#     return resized_image

# def update_model_preview(model_choice):
#     """Update the model preview image."""
#     return load_sample_image(SAMPLE_BBOXED_IMAGES, model_choice)

# def update_clothing_preview(clothing_choice):
#     """Update the clothing preview image."""
#     if clothing_choice == "None":
#         return None
#     return load_sample_image(SAMPLE_CLOTHING_IMAGES, clothing_choice)

# def update_mannequin_preview(mannequin_choice):
#     """Update the mannequin preview image."""
#     if mannequin_choice == "None":
#         return None
#     return load_sample_image(SAMPLE_MANNEQUIN_IMAGES, mannequin_choice)

# def generate_vton(
#     bboxed_model_choice: str,
#     clothing_upload: Optional[Image.Image],
#     clothing_sample: str,
#     num_steps: int,
#     text_guidance: float,
#     image_guidance: float,
#     negative_prompt: str,
#     seed: int
# ):
#     """Generate virtual try-on (Type 1)."""
#     try:
#         # Get bboxed model image
#         if bboxed_model_choice not in SAMPLE_BBOXED_IMAGES:
#             return None, "Please select a model from the dropdown"
        
#         bboxed_path = SAMPLE_BBOXED_IMAGES[bboxed_model_choice]
        
#         # Get clothing image (prioritize upload over sample)
#         clothing_img = None
#         if clothing_upload is not None:
#             clothing_img = clothing_upload
#         elif clothing_sample and clothing_sample != "None":
#             clothing_img = Image.open(SAMPLE_CLOTHING_IMAGES[clothing_sample])
#         else:
#             return None, "Please upload a clothing image or select a sample"
        
#         # Prepare images
#         bboxed_img = Image.open(bboxed_path)
#         input_images = preprocess_images([bboxed_img, clothing_img])
        
#         # Resize images
#         input_images = [resize_image(img) for img in input_images]
        
#         # Generate instruction
#         instruction = "Fill the black area in image1 with clothing in image2"
        
#         # Set random seed if requested
#         if seed == -1:
#             seed = random.randint(0, 100000)
        
#         generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        
#         # Generate image
#         results = pipeline(
#             prompt=instruction,
#             input_images=input_images,
#             num_inference_steps=num_steps,
#             max_sequence_length=1024,
#             text_guidance_scale=text_guidance,
#             image_guidance_scale=image_guidance,
#             negative_prompt=negative_prompt,
#             num_images_per_prompt=1,
#             generator=generator,
#             output_type="pil",
#         )
        
#         return results.images[0], f"Generated successfully with seed: {seed}"
    
#     except Exception as e:
#         return None, f"Error: {str(e)}"

# def generate_mannequin(
#     garment_type: str,
#     garment_upload: Optional[Image.Image],
#     garment_sample: str,
#     num_steps: int,
#     text_guidance: float,
#     image_guidance: float,
#     negative_prompt: str,
#     seed: int
# ):
#     """Generate ghost mannequin (Type 2)."""
#     try:
#         # Get garment image (prioritize upload over sample)
#         garment_img = None
#         if garment_upload is not None:
#             garment_img = garment_upload
#         elif garment_sample and garment_sample != "None":
#             garment_img = Image.open(SAMPLE_MANNEQUIN_IMAGES[garment_sample])
#         else:
#             return None, "Please upload a garment image or select a sample"
        
#         # Prepare image
#         input_images = preprocess_images([garment_img])
#         input_images = [resize_image(img) for img in input_images]
        
#         # Generate instruction
#         instruction = f"Create a ghost mannequin of the {garment_type}"
        
#         # Set random seed if requested
#         if seed == -1:
#             seed = random.randint(0, 100000)
        
#         generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        
#         # Generate image
#         results = pipeline(
#             prompt=instruction,
#             input_images=input_images,
#             num_inference_steps=num_steps,
#             max_sequence_length=1024,
#             text_guidance_scale=text_guidance,
#             image_guidance_scale=image_guidance,
#             negative_prompt=negative_prompt,
#             num_images_per_prompt=1,
#             generator=generator,
#             output_type="pil",
#         )
        
#         return results.images[0], f"Generated successfully with seed: {seed}"
    
#     except Exception as e:
#         return None, f"Error: {str(e)}"

# # Create Gradio interface
# with gr.Blocks(title="zyng Fashion Generator", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# 🎨zyng  Fashion Generator")
#     gr.Markdown("Generate virtual try-on images or ghost mannequin product shots")
    
#     with gr.Tabs():
#         # Type 1: Virtual Try-On
#         with gr.TabItem("👔 Virtual Try-On"):
#             gr.Markdown("### Fill clothing onto model with black box area")
            
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Step 1: Select Model")
#                     vton_model = gr.Dropdown(
#                         choices=list(SAMPLE_BBOXED_IMAGES.keys()),
#                         value=list(SAMPLE_BBOXED_IMAGES.keys())[0] if SAMPLE_BBOXED_IMAGES else None,
#                         label="Select Model (with black box area)"
#                     )
#                     vton_model_preview = gr.Image(
#                         label="Selected Model Preview",
#                         type="pil",
#                         interactive=False,
#                         height=300
#                     )
                
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Step 2: Choose Clothing")
#                     vton_clothing_upload = gr.Image(
#                         label="Upload Your Clothing Image",
#                         type="pil",
#                         height=300
#                     )
#                     gr.Markdown("**OR** select from samples:")
#                     vton_clothing_sample = gr.Dropdown(
#                         choices=["None"] + list(SAMPLE_CLOTHING_IMAGES.keys()),
#                         value="None",
#                         label="Sample Clothing"
#                     )
#                     vton_clothing_preview = gr.Image(
#                         label="Selected Sample Preview",
#                         type="pil",
#                         interactive=False,
#                         height=200
#                     )
                
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Generated Result")
#                     vton_output = gr.Image(label="Output", height=400)
#                     vton_status = gr.Textbox(label="Status", interactive=False)
            
#             gr.Markdown("### Generation Settings")
#             with gr.Row():
#                 with gr.Column():
#                     vton_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
#                     vton_text_guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Text Guidance Scale")
#                     vton_image_guidance = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Image Guidance Scale")
                
#                 with gr.Column():
#                     vton_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
#                     vton_negative = gr.Textbox(
#                         value=DEFAULT_NEGATIVE_PROMPT,
#                         label="Negative Prompt",
#                         lines=3
#                     )
            
#             vton_generate = gr.Button("🚀 Generate Virtual Try-On", variant="primary", size="lg")
        
#         # Type 2: Ghost Mannequin
#         with gr.TabItem("👻 Ghost Mannequin"):
#             gr.Markdown("### Create ghost mannequin product shots")
            
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Step 1: Configure Garment")
#                     mannequin_type = gr.Dropdown(
#                         choices=["jeans", "lower", "shirt", "upper", "dress", "jacket"],
#                         value="jeans",
#                         label="Garment Type"
#                     )
                    
#                     gr.Markdown("#### Step 2: Choose Garment Image")
#                     mannequin_upload = gr.Image(
#                         label="Upload Your Garment Image",
#                         type="pil",
#                         height=300
#                     )
#                     gr.Markdown("**OR** select from samples:")
#                     mannequin_sample = gr.Dropdown(
#                         choices=["None"] + list(SAMPLE_MANNEQUIN_IMAGES.keys()),
#                         value="None",
#                         label="Sample Garment"
#                     )
#                     mannequin_preview = gr.Image(
#                         label="Selected Sample Preview",
#                         type="pil",
#                         interactive=False,
#                         height=200
#                     )
                
#                 with gr.Column(scale=1):
#                     gr.Markdown("#### Generated Result")
#                     mannequin_output = gr.Image(label="Ghost Mannequin Output", height=400)
#                     mannequin_status = gr.Textbox(label="Status", interactive=False)
            
#             gr.Markdown("### Generation Settings")
#             with gr.Row():
#                 with gr.Column():
#                     mannequin_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
#                     mannequin_text_guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Text Guidance Scale")
#                     mannequin_image_guidance = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Image Guidance Scale")
                
#                 with gr.Column():
#                     mannequin_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
#                     mannequin_negative = gr.Textbox(
#                         value=DEFAULT_NEGATIVE_PROMPT,
#                         label="Negative Prompt",
#                         lines=3
#                     )
            
#             mannequin_generate = gr.Button("🚀 Generate Ghost Mannequin", variant="primary", size="lg")
    
#     # Sample Gallery
#     with gr.Row():
#         gr.Markdown("## 📸 Sample Gallery")
    
#     with gr.Row():
#         with gr.Column():
#             gr.Markdown("### Sample Models (Bboxed)")
#             model_gallery = gr.Gallery(
#                 value=[load_sample_image(SAMPLE_BBOXED_IMAGES, k) for k in SAMPLE_BBOXED_IMAGES.keys()],
#                 label="Available Models",
#                 show_label=False,
#                 elem_id="model_gallery",
#                 columns=3,
#                 rows=1,
#                 height=200
#             )
        
#         with gr.Column():
#             gr.Markdown("### Sample Clothing")
#             clothing_gallery = gr.Gallery(
#                 value=[load_sample_image(SAMPLE_CLOTHING_IMAGES, k) for k in SAMPLE_CLOTHING_IMAGES.keys()],
#                 label="Available Clothing",
#                 show_label=False,
#                 elem_id="clothing_gallery",
#                 columns=3,
#                 rows=1,
#                 height=200
#             )
        
#         with gr.Column():
#             gr.Markdown("### Sample Garments")
#             garment_gallery = gr.Gallery(
#                 value=[load_sample_image(SAMPLE_MANNEQUIN_IMAGES, k) for k in SAMPLE_MANNEQUIN_IMAGES.keys()],
#                 label="Available Garments",
#                 show_label=False,
#                 elem_id="garment_gallery",
#                 columns=3,
#                 rows=1,
#                 height=200
#             )
    
#     # Event handlers for updating previews
#     vton_model.change(
#         fn=update_model_preview,
#         inputs=[vton_model],
#         outputs=[vton_model_preview]
#     )
    
#     def update_clothing_preview_with_visibility(choice):
#         if choice == "None":
#             return None
#         return update_clothing_preview(choice)
    
#     def update_mannequin_preview_with_visibility(choice):
#         if choice == "None":
#             return None
#         return update_mannequin_preview(choice)
    
#     vton_clothing_sample.change(
#         fn=update_clothing_preview_with_visibility,
#         inputs=[vton_clothing_sample],
#         outputs=[vton_clothing_preview]
#     )
    
#     mannequin_sample.change(
#         fn=update_mannequin_preview_with_visibility,
#         inputs=[mannequin_sample],
#         outputs=[mannequin_preview]
#     )
    
#     # Load initial model preview
#     demo.load(
#         fn=update_model_preview,
#         inputs=[vton_model],
#         outputs=[vton_model_preview]
#     )
    
#     # Connect generation functions
#     vton_generate.click(
#         fn=generate_vton,
#         inputs=[
#             vton_model, vton_clothing_upload, vton_clothing_sample,
#             vton_steps, vton_text_guidance, vton_image_guidance,
#             vton_negative, vton_seed
#         ],
#         outputs=[vton_output, vton_status]
#     )
    
#     mannequin_generate.click(
#         fn=generate_mannequin,
#         inputs=[
#             mannequin_type, mannequin_upload, mannequin_sample,
#             mannequin_steps, mannequin_text_guidance, mannequin_image_guidance,
#             mannequin_negative, mannequin_seed
#         ],
#         outputs=[mannequin_output, mannequin_status]
#     )

# # Launch the app
# if __name__ == "__main__":
#     demo.launch(share=True, server_name="0.0.0.0", server_port=7860)



import dotenv
dotenv.load_dotenv(override=True)
import accelerate
import gradio as gr
import os
import glob
from typing import List, Tuple, Optional
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
import random

# Initialize accelerator and pipeline
accelerator = accelerate.Accelerator()
model_path = "OmniGen2/OmniGen2"

# Initialize pipeline once
pipeline = OmniGen2Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pipeline = pipeline.to(accelerator.device, dtype=torch.bfloat16)

# Load LoRA weights
pipeline.unload_lora_weights()
pipeline.load_lora_weights("experiments/vtonMixbig/checkpoint-100000/transformer_lora")

# Default negative prompt
DEFAULT_NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

# Function to dynamically discover models
def discover_front_models(base_path="vton"):
    """
    Discover all folders with 'front' in their name and check for bboxed_model.jpg
    Returns a dictionary with model names and their paths
    """
    models = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: {base_path} directory not found. Using fallback models.")
        # Fallback to original static list if vton directory doesn't exist
        return {
            "Model 1": "vton/262208_front/bboxed_model.jpg",
            "Model 2": "vton/276266_front/bboxed_model.jpg",
            "Model 3": "vton/269114_front/bboxed_model.jpg",
        }
    
    # Search for all directories containing 'front' in their name
    pattern = os.path.join(base_path, "*front*")
    front_dirs = glob.glob(pattern)
    
    for i, dir_path in enumerate(sorted(front_dirs), 1):
        if os.path.isdir(dir_path):
            # Check if bboxed_model.jpg exists in this directory
            bboxed_path = os.path.join(dir_path, "bboxed_model.jpg")
            if os.path.exists(bboxed_path):
                # Extract directory name for display
                dir_name = os.path.basename(dir_path)
                model_name = f"Model {i} ({dir_name})"
                models[model_name] = bboxed_path
                print(f"Found model: {model_name} -> {bboxed_path}")
    
    if not models:
        print("No valid models found. Using fallback models.")
        # Fallback if no valid models are found
        models = {
            "Model 1": "vton/262208_front/bboxed_model.jpg",
            "Model 2": "vton/276266_front/bboxed_model.jpg",
            "Model 3": "vton/269114_front/bboxed_model.jpg",
        }
    
    print(f"Total models discovered: {len(models)}")
    return models

# Discover models dynamically
SAMPLE_BBOXED_IMAGES = discover_front_models()

# Sample clothing images for Type 1
SAMPLE_CLOTHING_IMAGES = {
    "Clothing 1": "example_images/bf5e132d9b04e347365118427592e33e39ae5bad.jpg",
    "Clothing 2": "example_images/f3979db3a2dab4a04259b0bef35ae5719bb52bc8.jpg",
    "Clothing 3": "example_images/4442fbac4e3080ec20b2f14e353fea267249b0dd.jpg",
}

# Sample images for Type 2
SAMPLE_MANNEQUIN_IMAGES = {
    "Jeans Sample": "example_images/833435a34190ab6ee7d9b33f7e37d9345fcf9e17.jpg",
}

def load_sample_image(image_dict, key):
    """Load and return a sample image for display."""
    if key and key in image_dict:
        try:
            return Image.open(image_dict[key])
        except Exception as e:
            print(f"Error loading image {image_dict[key]}: {e}")
            return None
    return None

def preprocess_images(images: List) -> List[Image.Image]:
    """Preprocess input images."""
    processed = []
    for img in images:
        if img is not None:
            if isinstance(img, str):
                img = Image.open(img)
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            processed.append(img)
    return processed

def resize_image(image, max_size=1024):
    """Resize image if needed."""
    original_width, original_height = image.size
    
    if original_width <= max_size and original_height <= max_size:
        return image
    
    if original_width > original_height:
        new_width = max_size
        new_height = int((original_height * max_size) / original_width)
    else:
        new_height = max_size
        new_width = int((original_width * max_size) / original_height)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

def update_model_preview(model_choice):
    """Update the model preview image."""
    return load_sample_image(SAMPLE_BBOXED_IMAGES, model_choice)

def update_clothing_preview(clothing_choice):
    """Update the clothing preview image."""
    if clothing_choice == "None":
        return None
    return load_sample_image(SAMPLE_CLOTHING_IMAGES, clothing_choice)

def update_mannequin_preview(mannequin_choice):
    """Update the mannequin preview image."""
    if mannequin_choice == "None":
        return None
    return load_sample_image(SAMPLE_MANNEQUIN_IMAGES, mannequin_choice)

def refresh_models():
    """Refresh the list of available models."""
    global SAMPLE_BBOXED_IMAGES
    SAMPLE_BBOXED_IMAGES = discover_front_models()
    return gr.Dropdown.update(choices=list(SAMPLE_BBOXED_IMAGES.keys()), 
                             value=list(SAMPLE_BBOXED_IMAGES.keys())[0] if SAMPLE_BBOXED_IMAGES else None)

def generate_vton(
    bboxed_model_choice: str,
    clothing_upload: Optional[Image.Image],
    clothing_sample: str,
    num_steps: int,
    text_guidance: float,
    image_guidance: float,
    negative_prompt: str,
    seed: int
):
    """Generate virtual try-on (Type 1)."""
    try:
        # Get bboxed model image
        if bboxed_model_choice not in SAMPLE_BBOXED_IMAGES:
            return None, "Please select a model from the dropdown"
        
        bboxed_path = SAMPLE_BBOXED_IMAGES[bboxed_model_choice]
        
        # Check if the model file exists
        if not os.path.exists(bboxed_path):
            return None, f"Model file not found: {bboxed_path}"
        
        # Get clothing image (prioritize upload over sample)
        clothing_img = None
        if clothing_upload is not None:
            clothing_img = clothing_upload
        elif clothing_sample and clothing_sample != "None":
            clothing_img = Image.open(SAMPLE_CLOTHING_IMAGES[clothing_sample])
        else:
            return None, "Please upload a clothing image or select a sample"
        
        # Prepare images
        bboxed_img = Image.open(bboxed_path)
        input_images = preprocess_images([bboxed_img, clothing_img])
        
        # Resize images
        input_images = [resize_image(img) for img in input_images]
        
        # Generate instruction
        instruction = "Fill the black area in image1 with clothing in image2"
        
        # Set random seed if requested
        if seed == -1:
            seed = random.randint(0, 100000)
        
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        
        # Generate image
        results = pipeline(
            prompt=instruction,
            input_images=input_images,
            num_inference_steps=num_steps,
            max_sequence_length=1024,
            text_guidance_scale=text_guidance,
            image_guidance_scale=image_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        
        return results.images[0], f"Generated successfully with seed: {seed}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_mannequin(
    garment_type: str,
    garment_upload: Optional[Image.Image],
    garment_sample: str,
    num_steps: int,
    text_guidance: float,
    image_guidance: float,
    negative_prompt: str,
    seed: int
):
    """Generate ghost mannequin (Type 2)."""
    try:
        # Get garment image (prioritize upload over sample)
        garment_img = None
        if garment_upload is not None:
            garment_img = garment_upload
        elif garment_sample and garment_sample != "None":
            garment_img = Image.open(SAMPLE_MANNEQUIN_IMAGES[garment_sample])
        else:
            return None, "Please upload a garment image or select a sample"
        
        # Prepare image
        input_images = preprocess_images([garment_img])
        input_images = [resize_image(img) for img in input_images]
        
        # Generate instruction
        instruction = f"Create a ghost mannequin of the {garment_type}"
        
        # Set random seed if requested
        if seed == -1:
            seed = random.randint(0, 100000)
        
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        
        # Generate image
        results = pipeline(
            prompt=instruction,
            input_images=input_images,
            num_inference_steps=num_steps,
            max_sequence_length=1024,
            text_guidance_scale=text_guidance,
            image_guidance_scale=image_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        
        return results.images[0], f"Generated successfully with seed: {seed}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="zyng Fashion Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨zyng Fashion Generator")
    gr.Markdown("Generate virtual try-on images or ghost mannequin product shots")
    
    with gr.Tabs():
        # Type 1: Virtual Try-On
        with gr.TabItem("👔 Virtual Try-On"):
            gr.Markdown("### Fill clothing onto model with black box area")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Select Model")
                    with gr.Row():
                        vton_model = gr.Dropdown(
                            choices=list(SAMPLE_BBOXED_IMAGES.keys()),
                            value=list(SAMPLE_BBOXED_IMAGES.keys())[0] if SAMPLE_BBOXED_IMAGES else None,
                            label="Select Model (with black box area)",
                            scale=4
                        )
                        refresh_btn = gr.Button("🔄 Refresh Models", scale=1, size="sm")
                    
                    vton_model_preview = gr.Image(
                        label="Selected Model Preview",
                        type="pil",
                        interactive=False,
                        height=300
                    )
                    
                    # Display model count
                    model_count = gr.Markdown(f"**{len(SAMPLE_BBOXED_IMAGES)} models available**")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 2: Choose Clothing")
                    vton_clothing_upload = gr.Image(
                        label="Upload Your Clothing Image",
                        type="pil",
                        height=300
                    )
                    gr.Markdown("**OR** select from samples:")
                    vton_clothing_sample = gr.Dropdown(
                        choices=["None"] + list(SAMPLE_CLOTHING_IMAGES.keys()),
                        value="None",
                        label="Sample Clothing"
                    )
                    vton_clothing_preview = gr.Image(
                        label="Selected Sample Preview",
                        type="pil",
                        interactive=False,
                        height=200
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Generated Result")
                    vton_output = gr.Image(label="Output", height=400)
                    vton_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Generation Settings")
            with gr.Row():
                with gr.Column():
                    vton_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
                    vton_text_guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Text Guidance Scale")
                    vton_image_guidance = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Image Guidance Scale")
                
                with gr.Column():
                    vton_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    vton_negative = gr.Textbox(
                        value=DEFAULT_NEGATIVE_PROMPT,
                        label="Negative Prompt",
                        lines=3
                    )
            
            vton_generate = gr.Button("🚀 Generate Virtual Try-On", variant="primary", size="lg")
        
        # Type 2: Ghost Mannequin
        with gr.TabItem("👻 Ghost Mannequin"):
            gr.Markdown("### Create ghost mannequin product shots")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Configure Garment")
                    mannequin_type = gr.Dropdown(
                        choices=["jeans", "lower", "shirt", "upper", "dress", "jacket"],
                        value="jeans",
                        label="Garment Type"
                    )
                    
                    gr.Markdown("#### Step 2: Choose Garment Image")
                    mannequin_upload = gr.Image(
                        label="Upload Your Garment Image",
                        type="pil",
                        height=300
                    )
                    gr.Markdown("**OR** select from samples:")
                    mannequin_sample = gr.Dropdown(
                        choices=["None"] + list(SAMPLE_MANNEQUIN_IMAGES.keys()),
                        value="None",
                        label="Sample Garment"
                    )
                    mannequin_preview = gr.Image(
                        label="Selected Sample Preview",
                        type="pil",
                        interactive=False,
                        height=200
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Generated Result")
                    mannequin_output = gr.Image(label="Ghost Mannequin Output", height=400)
                    mannequin_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Generation Settings")
            with gr.Row():
                with gr.Column():
                    mannequin_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
                    mannequin_text_guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Text Guidance Scale")
                    mannequin_image_guidance = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Image Guidance Scale")
                
                with gr.Column():
                    mannequin_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    mannequin_negative = gr.Textbox(
                        value=DEFAULT_NEGATIVE_PROMPT,
                        label="Negative Prompt",
                        lines=3
                    )
            
            mannequin_generate = gr.Button("🚀 Generate Ghost Mannequin", variant="primary", size="lg")
    
    # Sample Gallery
    with gr.Row():
        gr.Markdown("## 📸 Sample Gallery")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Sample Models (Bboxed)")
            model_gallery = gr.Gallery(
                value=[load_sample_image(SAMPLE_BBOXED_IMAGES, k) for k in list(SAMPLE_BBOXED_IMAGES.keys())[:6]],  # Show first 6 models
                label="Available Models",
                show_label=False,
                elem_id="model_gallery",
                columns=3,
                rows=2,
                height=300
            )
        
        with gr.Column():
            gr.Markdown("### Sample Clothing")
            clothing_gallery = gr.Gallery(
                value=[load_sample_image(SAMPLE_CLOTHING_IMAGES, k) for k in SAMPLE_CLOTHING_IMAGES.keys()],
                label="Available Clothing",
                show_label=False,
                elem_id="clothing_gallery",
                columns=3,
                rows=1,
                height=200
            )
        
        with gr.Column():
            gr.Markdown("### Sample Garments")
            garment_gallery = gr.Gallery(
                value=[load_sample_image(SAMPLE_MANNEQUIN_IMAGES, k) for k in SAMPLE_MANNEQUIN_IMAGES.keys()],
                label="Available Garments",
                show_label=False,
                elem_id="garment_gallery",
                columns=3,
                rows=1,
                height=200
            )
    
    # Event handlers for refresh button
    def refresh_models_and_update():
        global SAMPLE_BBOXED_IMAGES
        SAMPLE_BBOXED_IMAGES = discover_front_models()
        new_choices = list(SAMPLE_BBOXED_IMAGES.keys())
        new_value = new_choices[0] if new_choices else None
        model_count_text = f"**{len(SAMPLE_BBOXED_IMAGES)} models available**"
        
        # Update gallery with new models (show first 6)
        gallery_images = [load_sample_image(SAMPLE_BBOXED_IMAGES, k) for k in new_choices[:6]]
        
        return (
            gr.Dropdown.update(choices=new_choices, value=new_value),
            model_count_text,
            gallery_images
        )
    
    refresh_btn.click(
        fn=refresh_models_and_update,
        outputs=[vton_model, model_count, model_gallery]
    )
    
    # Event handlers for updating previews
    vton_model.change(
        fn=update_model_preview,
        inputs=[vton_model],
        outputs=[vton_model_preview]
    )
    
    def update_clothing_preview_with_visibility(choice):
        if choice == "None":
            return None
        return update_clothing_preview(choice)
    
    def update_mannequin_preview_with_visibility(choice):
        if choice == "None":
            return None
        return update_mannequin_preview(choice)
    
    vton_clothing_sample.change(
        fn=update_clothing_preview_with_visibility,
        inputs=[vton_clothing_sample],
        outputs=[vton_clothing_preview]
    )
    
    mannequin_sample.change(
        fn=update_mannequin_preview_with_visibility,
        inputs=[mannequin_sample],
        outputs=[mannequin_preview]
    )
    
    # Load initial model preview
    demo.load(
        fn=update_model_preview,
        inputs=[vton_model],
        outputs=[vton_model_preview]
    )
    
    # Connect generation functions
    vton_generate.click(
        fn=generate_vton,
        inputs=[
            vton_model, vton_clothing_upload, vton_clothing_sample,
            vton_steps, vton_text_guidance, vton_image_guidance,
            vton_negative, vton_seed
        ],
        outputs=[vton_output, vton_status]
    )
    
    mannequin_generate.click(
        fn=generate_mannequin,
        inputs=[
            mannequin_type, mannequin_upload, mannequin_sample,
            mannequin_steps, mannequin_text_guidance, mannequin_image_guidance,
            mannequin_negative, mannequin_seed
        ],
        outputs=[mannequin_output, mannequin_status]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


    