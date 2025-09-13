
import torch
from typing import Optional, Union, List

import os
import random
import glob
from PIL import Image


from torchvision import transforms

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor


def resize_image(image, max_size=1024):
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # If image is already smaller than max_size, return as-is
    if original_width <= max_size and original_height <= max_size:
        return image
    
    # Calculate the scaling factor
    if original_width > original_height:
        # Width is the limiting dimension
        new_width = max_size
        new_height = int((original_height * max_size) / original_width)
    else:
        # Height is the limiting dimension
        new_height = max_size
        new_width = int((original_width * max_size) / original_height)
    
    # Resize using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

    
class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

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
        input_folder="dataset",
        output_folder="nykaa_images",
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels

        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        # Collect data from folder structure
        self.data = self._collect_data_from_folders()
        self.tokenizer = tokenizer
        
    def _collect_data_from_folders(self):
        """
        Collect data from input and output folders.
        Creates triplets of (blacked_image, cropped_image, output_image).
        """
        data_items = []
        
        # Get all unique ID folders from input directory
        input_subfolders = [f for f in os.listdir(self.input_folder) 
                           if os.path.isdir(os.path.join(self.input_folder, f))]
        
        for unique_id in input_subfolders:
            input_subfolder = os.path.join(self.input_folder, unique_id)
            output_subfolder = os.path.join(self.output_folder, unique_id)
            
            # Skip if corresponding output folder doesn't exist
            if not os.path.exists(output_subfolder):
                continue
                
            # Get all images in input subfolder
            input_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                input_images.extend(glob.glob(os.path.join(input_subfolder, ext)))
                input_images.extend(glob.glob(os.path.join(input_subfolder, ext.upper())))
            
            # Get all images in output subfolder
            output_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                output_images.extend(glob.glob(os.path.join(output_subfolder, ext)))
                output_images.extend(glob.glob(os.path.join(output_subfolder, ext.upper())))
            
            # Group input images by prefix
            blacked_images = {}  # prefix -> image_path
            cropped_images = {}  # prefix -> image_path
            
            for img_path in input_images:
                filename = os.path.basename(img_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                if name_without_ext.endswith('_blacked'):
                    prefix = name_without_ext[:-8]  # Remove '_blacked'
                    blacked_images[prefix] = img_path
                elif name_without_ext.endswith('_cropped'):
                    prefix = name_without_ext[:-8]  # Remove '_cropped'
                    cropped_images[prefix] = img_path
            
            # Group output images by prefix (filename without extension)
            output_by_prefix = {}
            for img_path in output_images:
                filename = os.path.basename(img_path)
                prefix = os.path.splitext(filename)[0]
                output_by_prefix[prefix] = img_path
            
            # Create triplets: for each blacked image, find matching output and any different cropped
            for blacked_prefix, blacked_path in blacked_images.items():
                # Find matching output image
                if blacked_prefix not in output_by_prefix:
                    continue
                output_path = output_by_prefix[blacked_prefix]
                
                # Find cropped images with different prefixes
                available_cropped = [path for prefix, path in cropped_images.items() 
                                   if prefix != blacked_prefix]
                
                if not available_cropped:
                    continue
                
                # Create one data item for each available cropped image
                for cropped_path in available_cropped:
                    data_item = {
                        'task_type': 'image_inpainting',
                        'instruction': 'fill the black part in first image with the person in the second image',
                        'input_images': [blacked_path, cropped_path],
                        'output_image': output_path,
                        'unique_id': unique_id,
                        'blacked_prefix': blacked_prefix,
                        'cropped_prefix': os.path.splitext(os.path.basename(cropped_path))[0][:-8]  # Remove '_cropped'
                    }
                    data_items.append(data_item)
        
        print(f"Collected {len(data_items)} training samples from {len(input_subfolders)} unique ID folders")
        return data_items
    
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
    
    def process_item(self, data_item):
        assert data_item['instruction'] is not None

        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(data_item['instruction'], self.SYSTEM_PROMPT)

        if not drop_ref_img and 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []

            max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image=resize_image(input_image)
                input_image = self.image_processor.preprocess(input_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)
                
                input_images.append(input_image)
        else:
            input_images_path, input_images = None, None

        output_image_path = data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image=resize_image(output_image)
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'unique_id': data_item.get('unique_id', ''),
            'blacked_prefix': data_item.get('blacked_prefix', ''),
            'cropped_prefix': data_item.get('cropped_prefix', ''),
        }
        return data

    def __getitem__(self, index):
        max_retries = 12

        current_index = index
        for attempt in range(max_retries):
            try:
                data_item = self.data[current_index]
                return self.process_item(data_item)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.data) - 1)
                    continue
        
    def __len__(self):
        return len(self.data)

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