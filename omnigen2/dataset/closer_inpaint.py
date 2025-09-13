from typing import Optional, Union, List

import os
import random
import numpy as np
from PIL import Image
from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor
import torch
from torchvision import transforms

class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality virtual try-on images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."
    instruction_templates="Fill the black area in image1 with clothing in image2"
    def __init__(
        self,
        tokenizer,
        use_chat_template: bool,
        max_input_pixels: Optional[Union[int, List[int]]] = None,
        max_output_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
        gray_value: int = 128,
        dataset_dir= "./simple_data",
    ):
        self.dataset_dir = dataset_dir
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.gray_value = gray_value
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.use_chat_template = use_chat_template
        
        self.tokenizer = tokenizer
        
        # Default instruction templates
       
        
        # Collect all sample folders
        self.sample_folders = self._collect_sample_folders()
        
    def _collect_sample_folders(self):
        """Collect all valid sample folders"""
        sample_folders = []
        
        for folder_name in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            # Check if required files exist
            model_path = os.path.join(folder_path, 'model.jpg')
            garment_path = os.path.join(folder_path, 'garment.jpg')
            mask_path = os.path.join(folder_path, 'mask.png')
            
            if all(os.path.exists(p) for p in [model_path, garment_path, mask_path]):
                sample_folders.append(folder_path)
        
        print(f"Found {len(sample_folders)} valid sample folders")
        return sample_folders
    
    def _create_masked_model_image(self, model_image, mask_image):
        """Create model image with masked areas grayed out"""
        # Convert images to numpy arrays
        model_array = np.array(model_image)
        mask_array = np.array(mask_image)
        
        # Normalize mask to 0-1 range
        mask_normalized = mask_array.astype(np.float32) / 255.0
        
        # Create gray image
        gray_image = np.full_like(model_array, self.gray_value)
        
        # Blend: where mask is white (1), use gray; where mask is black (0), use original
        # Expand mask to 3 channels for RGB
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
        # Apply mask: masked areas become gray, unmasked areas stay original
        masked_model = model_array * (1 - mask_3d) + gray_image * mask_3d
        
        # Convert back to PIL Image
        return Image.fromarray(masked_model.astype(np.uint8))
    
    def _get_category_from_folder(self, folder_path):
        """Extract category from folder name"""
        folder_name = os.path.basename(folder_path).lower()
        
        if 'upper_body' in folder_name:
            return 'upper body clothing'
        elif 'lower_body' in folder_name:
            return 'lower body clothing'
        elif 'dresses' in folder_name:
            return 'dress'
        else:
            return 'clothing'
    
    def _generate_instruction(self, folder_path):
        """Generate instruction for the sample"""
        category = self._get_category_from_folder(folder_path)
        template = self.instruction_templates
        return template.format(category=category)
    
    def apply_chat_template(self, instruction, system_prompt):
        if self.use_chat_template:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction
    
    def process_item(self, folder_path):
        """Process a single sample"""
        # Load images
        model_path = os.path.join(folder_path, 'model.jpg')
        garment_path = os.path.join(folder_path, 'garment.jpg')
        mask_path = os.path.join(folder_path, 'mask.png')
        
        model_image = Image.open(model_path).convert('RGB')
        garment_image = Image.open(garment_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        
        # Ensure mask is same size as model image
        if mask_image.size != model_image.size:
            mask_image = mask_image.resize(model_image.size, Image.LANCZOS)
        
        # Create masked model image
        masked_model_image = self._create_masked_model_image(model_image, mask_image)
        
        # Handle dropouts
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
        # Generate instruction
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction_text = self._generate_instruction(folder_path)
            instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
        # Process input images
        if not drop_ref_img:
            input_images = []
            input_images_path = [model_path, garment_path]  # Store paths for reference
            
            max_input_pixels = self.max_input_pixels[1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
            # Process masked model image
            processed_masked_model = self.image_processor.preprocess(
                masked_model_image, 
                max_pixels=max_input_pixels, 
                max_side_length=self.max_side_length
            )
            input_images.append(processed_masked_model)
            
            # Process garment image
            processed_garment = self.image_processor.preprocess(
                garment_image, 
                max_pixels=max_input_pixels, 
                max_side_length=self.max_side_length
            )
            input_images.append(processed_garment)
        else:
            input_images_path, input_images = None, None
        
        # Process output image
        output_image = self.image_processor.preprocess(
            model_image, 
            max_pixels=self.max_output_pixels, 
            max_side_length=self.max_side_length
        )
        
        data = {
            'task_type': 'virtual_try_on',
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': model_path,
        }
        return data

    def __getitem__(self, index):
        max_retries = 12
        
        current_index = index % len(self.sample_folders)
        for attempt in range(max_retries):
            try:
                folder_path = self.sample_folders[current_index]
                return self.process_item(folder_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.sample_folders) - 1)
                    continue
        
    def __len__(self):
        return len(self.sample_folders)


class OmniGen2Collator():
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        task_type = [data['task_type'] for data in batch]
        instruction = [data['instruction'] for data in batch]
        input_images_path = [data['input_images_path'] for data in batch]
        input_images = [data['input_images'] for data in batch]
        output_image = [data['output_image'] for data in batch]
        output_image_path = [data['output_image_path'] for data in batch]

        text_inputs = self.tokenizer(
            instruction,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        data = {
            "task_type": task_type,
            "text_ids": text_inputs.input_ids,
            "text_mask": text_inputs.attention_mask,
            "input_images": input_images, 
            "input_images_path": input_images_path,
            "output_image": output_image,
            "output_image_path": output_image_path,
        }
        return data