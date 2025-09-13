# from typing import Optional, Union, List

# import os
# import random
# import numpy as np
# from PIL import Image
# from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor
# import torch
# from torchvision import transforms

# class OmniGen2TrainDataset(torch.utils.data.Dataset):
#     SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality virtual try-on images based on user instructions."
#     SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."
    
#     # Updated instruction templates for different colors
#     instruction_templates = [
#         "Fill the grey area in image1 with clothing in image2",
#         "Fill the black area in image1 with clothing in image2", 
#         "Fill the blue area in image1 with clothing in image2"
#     ]
    
#     # Color mapping
#     MASK_COLORS = {
#         'grey': 128,
#         'black': 0,
#         'blue': [0, 0, 255]  # RGB values for blue
#     }
    
#     COLOR_NAMES = ['grey', 'black', 'blue']
    
#     def __init__(
#         self,
#         tokenizer,
#         use_chat_template: bool,
#         max_input_pixels: Optional[Union[int, List[int]]] = None,
#         max_output_pixels: Optional[int] = None,
#         max_side_length: Optional[int] = None,
#         img_scale_num: int = 16,
#         prompt_dropout_prob: float = 0.0,
#         ref_img_dropout_prob: float = 0.0,
#         # Remove single gray_value, we'll use MASK_COLORS instead
#         dataset_dir= "./Simple_data3",
#         # New cropping parameters
#         crop_prob: float = 0.7,  # Probability of cropping (0.0 = never crop, 1.0 = always crop)
#         min_padding: int = 25,   # Minimum padding around mask
#         max_padding: int = 100,  # Maximum padding around mask
#         min_crop_size: int = 256, # Minimum crop size to ensure reasonable image size
#         # Color distribution weights (optional)
#         color_weights: Optional[List[float]] = None,  # [grey_weight, black_weight, blue_weight]
#     ):
#         self.dataset_dir = dataset_dir
#         self.max_input_pixels = max_input_pixels
#         self.max_output_pixels = max_output_pixels
#         self.max_side_length = max_side_length
#         self.prompt_dropout_prob = prompt_dropout_prob
#         self.ref_img_dropout_prob = ref_img_dropout_prob
        
#         # Set default color weights if not provided (equal probability)
#         self.color_weights = color_weights if color_weights else [1.0, 1.0, 1.0]
        
#         self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
#         self.use_chat_template = use_chat_template
        
#         # Cropping parameters
#         self.crop_prob = crop_prob
#         self.min_padding = min_padding
#         self.max_padding = max_padding
#         self.min_crop_size = min_crop_size
        
#         self.tokenizer = tokenizer
        
#         # Collect all sample folders
#         self.sample_folders = self._collect_sample_folders()
        
#     def _collect_sample_folders(self):
#         """Collect all valid sample folders"""
#         sample_folders = []
        
#         for folder_name in os.listdir(self.dataset_dir):
#             folder_path = os.path.join(self.dataset_dir, folder_name)
            
#             if not os.path.isdir(folder_path):
#                 continue
                
#             # Check if required files exist
#             model_path = os.path.join(folder_path, 'model.jpg')
#             garment_path = os.path.join(folder_path, 'garment.jpg')
#             mask_path = os.path.join(folder_path, 'mask.png')
            
#             if all(os.path.exists(p) for p in [model_path, garment_path, mask_path]):
#                 sample_folders.append(folder_path)
        
#         print(f"Found {len(sample_folders)} valid sample folders")
#         return sample_folders
    
#     def _select_random_color(self):
#         """Select a random color based on weights"""
#         return random.choices(self.COLOR_NAMES, weights=self.color_weights, k=1)[0]
    
#     def _get_mask_bbox(self, mask_image):
#         """Get bounding box of the mask area"""
#         mask_array = np.array(mask_image)
        
#         # Find non-zero pixels (mask areas)
#         coords = np.column_stack(np.where(mask_array > 0))
        
#         if len(coords) == 0:
#             # If no mask found, return None
#             return None
        
#         # Get bounding box coordinates
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
        
#         return (x_min, y_min, x_max, y_max)
    
#     def _apply_random_crop(self, image, mask_image):
#         """Apply random cropping around mask area with random padding"""
#         # Get mask bounding box
#         bbox = self._get_mask_bbox(mask_image)
        
#         if bbox is None:
#             # No mask found, return original image
#             return image, mask_image
        
#         x_min, y_min, x_max, y_max = bbox
#         img_width, img_height = image.size
        
#         # Generate random padding
#         padding = random.randint(self.min_padding, self.max_padding)
        
#         # Calculate crop bounds with padding
#         crop_x_min = max(0, x_min - padding)
#         crop_y_min = max(0, y_min - padding)
#         crop_x_max = min(img_width, x_max + padding)
#         crop_y_max = min(img_height, y_max + padding)
        
#         # Ensure minimum crop size
#         crop_width = crop_x_max - crop_x_min
#         crop_height = crop_y_max - crop_y_min
        
#         # Perform the crop
#         crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
#         cropped_image = image.crop(crop_box)
#         cropped_mask = mask_image.crop(crop_box)
        
#         return cropped_image, cropped_mask
    
#     def _create_masked_model_image(self, model_image, mask_image, color_name):
#         """Create model image with masked areas filled with specified color"""
#         # Convert images to numpy arrays
#         model_array = np.array(model_image)
#         mask_array = np.array(mask_image)
        
#         # Normalize mask to 0-1 range
#         mask_normalized = mask_array.astype(np.float32) / 255.0
        
#         # Create colored image based on selected color
#         color_value = self.MASK_COLORS[color_name]
        
#         if color_name == 'blue':
#             # For blue, use RGB values
#             color_image = np.full_like(model_array, color_value)
#         else:
#             # For grey and black, use single value for all channels
#             color_image = np.full_like(model_array, color_value)
        
#         # Blend: where mask is white (1), use color; where mask is black (0), use original
#         # Expand mask to 3 channels for RGB
#         mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
#         # Apply mask: masked areas become colored, unmasked areas stay original
#         masked_model = model_array * (1 - mask_3d) + color_image * mask_3d
        
#         # Convert back to PIL Image
#         return Image.fromarray(masked_model.astype(np.uint8))
    
#     def _get_category_from_folder(self, folder_path):
#         """Extract category from folder name"""
#         folder_name = os.path.basename(folder_path).lower()
        
#         if 'upper_body' in folder_name:
#             return 'upper body clothing'
#         elif 'lower_body' in folder_name:
#             return 'lower body clothing'
#         elif 'dresses' in folder_name:
#             return 'dress'
#         else:
#             return 'clothing'
    
#     def _generate_instruction(self, folder_path, color_name):
#         """Generate instruction for the sample with specified color"""
#         category = self._get_category_from_folder(folder_path)
        
#         # Select the appropriate template based on color
#         color_index = self.COLOR_NAMES.index(color_name)
#         template = self.instruction_templates[color_index]
        
#         return template.format(category=category)
    
#     def apply_chat_template(self, instruction, system_prompt):
#         if self.use_chat_template:
#             prompt = [
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {"role": "user", "content": instruction},
#             ]
#             instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
#         return instruction
    
#     def process_item(self, folder_path):
#         """Process a single sample"""
#         # Load images
#         model_path = os.path.join(folder_path, 'model.jpg')
#         if random.random() >0.95:
#             garment_path = os.path.join(folder_path, 'garment.jpg')
#         else:
#             garment_path = os.path.join(folder_path, 'clipped_garment.png')
#         mask_path = os.path.join(folder_path, 'mask.png')
        
#         model_image = Image.open(model_path).convert('RGB')
#         garment_image = Image.open(garment_path).convert('RGB')
#         mask_image = Image.open(mask_path).convert('L')
        
#         # Ensure mask is same size as model image
#         if mask_image.size != model_image.size:
#             mask_image = mask_image.resize(model_image.size, Image.LANCZOS)
        
#         # Decide whether to crop or not
#         should_crop = random.random() < self.crop_prob
        
#         if should_crop:
#             # Apply random cropping to model image and mask
#             model_image, mask_image = self._apply_random_crop(model_image, mask_image)
#             # Note: garment image is not cropped as it should remain full
        
#         # Select random color for this sample
#         selected_color = self._select_random_color()
        
#         # Create masked model image with selected color (after potential cropping)
#         masked_model_image = self._create_masked_model_image(model_image, mask_image, selected_color)
        
#         # Handle dropouts
#         drop_prompt = random.random() < self.prompt_dropout_prob
#         drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
#         # Generate instruction with selected color
#         if drop_prompt:
#             instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
#         else:
#             instruction_text = self._generate_instruction(folder_path, selected_color)
#             instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
#         # Process input images
#         if not drop_ref_img:
#             input_images = []
#             input_images_path = [model_path, garment_path]  # Store paths for reference
            
#             max_input_pixels = self.max_input_pixels[1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
#             # Process masked model image (potentially cropped)
#             processed_masked_model = self.image_processor.preprocess(
#                 masked_model_image, 
#                 max_pixels=max_input_pixels, 
#                 max_side_length=self.max_side_length
#             )
#             input_images.append(processed_masked_model)
            
#             # Process garment image (always full image)
#             processed_garment = self.image_processor.preprocess(
#                 garment_image, 
#                 max_pixels=max_input_pixels, 
#                 max_side_length=self.max_side_length
#             )
#             input_images.append(processed_garment)
#         else:
#             input_images_path, input_images = None, None
        
#         # Process output image (use cropped version if cropping was applied)
#         output_image = self.image_processor.preprocess(
#             model_image, 
#             max_pixels=self.max_output_pixels, 
#             max_side_length=self.max_side_length
#         )
        
#         data = {
#             'task_type': 'virtual_try_on',
#             'instruction': instruction,
#             'input_images_path': input_images_path,
#             'input_images': input_images,
#             'output_image': output_image,
#             'output_image_path': model_path,
#             'is_cropped': should_crop,  # Optional: track whether cropping was applied
#             'mask_color': selected_color,  # Track which color was used
#         }
#         return data

#     def __getitem__(self, index):
#         max_retries = 12
        
#         current_index = index % len(self.sample_folders)
#         for attempt in range(max_retries):
#             try:
#                 folder_path = self.sample_folders[current_index]
#                 return self.process_item(folder_path)
#             except Exception as e:
#                 if attempt == max_retries - 1:
#                     raise e
#                 else:
#                     # Try a different index for the next attempt
#                     current_index = random.randint(0, len(self.sample_folders) - 1)
#                     continue
        
#     def __len__(self):
#         return len(self.sample_folders)


# class OmniGen2Collator():
#     def __init__(self, tokenizer, max_token_len):
#         self.tokenizer = tokenizer
#         self.max_token_len = max_token_len

#     def __call__(self, batch):
#         task_type = [data['task_type'] for data in batch]
#         instruction = [data['instruction'] for data in batch]
#         input_images_path = [data['input_images_path'] for data in batch]
#         input_images = [data['input_images'] for data in batch]
#         output_image = [data['output_image'] for data in batch]
#         output_image_path = [data['output_image_path'] for data in batch]

#         text_inputs = self.tokenizer(
#             instruction,
#             padding="longest",
#             max_length=self.max_token_len,
#             truncation=True,
#             return_tensors="pt",
#         )

#         data = {
#             "task_type": task_type,
#             "text_ids": text_inputs.input_ids,
#             "text_mask": text_inputs.attention_mask,
#             "input_images": input_images, 
#             "input_images_path": input_images_path,
#             "output_image": output_image,
#             "output_image_path": output_image_path,
#         }
#         return data



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
    
    # Updated instruction templates for different colors (forward task)
    instruction_templates = [
        "Fill the grey area in image1 with clothing in image2",
        "Fill the black area in image1 with clothing in image2", 
        "Fill the blue area in image1 with clothing in image2"
    ]
    
    # New reverse task instruction templates
    reverse_instruction_templates = [
        "Create a flat lay image of the clothing item worn by the person in image1",
        "Generate a flat lay shot of the garment from the person wearing it in image1", 
        "Extract and create a flat lay representation of the clothing from image1"
    ]
    
    # Color mapping
    MASK_COLORS = {
        'grey': 128,
        'black': 0,
        'blue': [0, 0, 255]  # RGB values for blue
    }
    
    COLOR_NAMES = ['grey', 'black', 'blue']
    
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
        # Remove single gray_value, we'll use MASK_COLORS instead
        dataset_dir= "./Simple_data3",
        # New cropping parameters
        crop_prob: float = 0.7,  # Probability of cropping (0.0 = never crop, 1.0 = always crop)
        min_padding: int = 25,   # Minimum padding around mask
        max_padding: int = 100,  # Maximum padding around mask
        min_crop_size: int = 256, # Minimum crop size to ensure reasonable image size
        # Color distribution weights (optional)
        color_weights: Optional[List[float]] = None,  # [grey_weight, black_weight, blue_weight]
        # New parameter for reverse task probability
        reverse_task_prob: float = 0.5,  # 50% probability for reverse task
    ):
        self.dataset_dir = dataset_dir
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.reverse_task_prob = reverse_task_prob
        
        # Set default color weights if not provided (equal probability)
        self.color_weights = color_weights if color_weights else [1.0, 1.0, 1.0]
        
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.use_chat_template = use_chat_template
        
        # Cropping parameters
        self.crop_prob = crop_prob
        self.min_padding = min_padding
        self.max_padding = max_padding
        self.min_crop_size = min_crop_size
        
        self.tokenizer = tokenizer
        
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
    
    def _select_random_color(self):
        """Select a random color based on weights"""
        return random.choices(self.COLOR_NAMES, weights=self.color_weights, k=1)[0]
    
    def _get_mask_bbox(self, mask_image):
        """Get bounding box of the mask area"""
        mask_array = np.array(mask_image)
        
        # Find non-zero pixels (mask areas)
        coords = np.column_stack(np.where(mask_array > 0))
        
        if len(coords) == 0:
            # If no mask found, return None
            return None
        
        # Get bounding box coordinates
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (x_min, y_min, x_max, y_max)
    
    def _apply_random_crop(self, image, mask_image):
        """Apply random cropping around mask area with random padding"""
        # Get mask bounding box
        bbox = self._get_mask_bbox(mask_image)
        
        if bbox is None:
            # No mask found, return original image
            return image, mask_image
        
        x_min, y_min, x_max, y_max = bbox
        img_width, img_height = image.size
        
        # Generate random padding
        padding = random.randint(self.min_padding, self.max_padding)
        
        # Calculate crop bounds with padding
        crop_x_min = max(0, x_min - padding)
        crop_y_min = max(0, y_min - padding)
        crop_x_max = min(img_width, x_max + padding)
        crop_y_max = min(img_height, y_max + padding)
        
        # Ensure minimum crop size
        crop_width = crop_x_max - crop_x_min
        crop_height = crop_y_max - crop_y_min
        
        # Perform the crop
        crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        cropped_image = image.crop(crop_box)
        cropped_mask = mask_image.crop(crop_box)
        
        return cropped_image, cropped_mask
    
    def _create_masked_model_image(self, model_image, mask_image, color_name):
        """Create model image with masked areas filled with specified color"""
        # Convert images to numpy arrays
        model_array = np.array(model_image)
        mask_array = np.array(mask_image)
        
        # Normalize mask to 0-1 range
        mask_normalized = mask_array.astype(np.float32) / 255.0
        
        # Create colored image based on selected color
        color_value = self.MASK_COLORS[color_name]
        
        if color_name == 'blue':
            # For blue, use RGB values
            color_image = np.full_like(model_array, color_value)
        else:
            # For grey and black, use single value for all channels
            color_image = np.full_like(model_array, color_value)
        
        # Blend: where mask is white (1), use color; where mask is black (0), use original
        # Expand mask to 3 channels for RGB
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
        # Apply mask: masked areas become colored, unmasked areas stay original
        masked_model = model_array * (1 - mask_3d) + color_image * mask_3d
        
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
    
    def _generate_instruction(self, folder_path, color_name, is_reverse=False):
        """Generate instruction for the sample with specified color"""
        category = self._get_category_from_folder(folder_path)
        
        if is_reverse:
            # For reverse task, randomly select from reverse templates
            template = random.choice(self.reverse_instruction_templates)
            return template.format(category=category)
        else:
            # Select the appropriate template based on color for forward task
            color_index = self.COLOR_NAMES.index(color_name)
            template = self.instruction_templates[color_index]
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
        # Determine if this is a reverse task
        is_reverse_task = random.random() < self.reverse_task_prob
        
        # Load images
        model_path = os.path.join(folder_path, 'model.jpg')
        if random.random() > 0.95:
            garment_path = os.path.join(folder_path, 'garment.jpg')
        else:
            garment_path = os.path.join(folder_path, 'clipped_garment.png')
        mask_path = os.path.join(folder_path, 'mask.png')
        
        model_image = Image.open(model_path).convert('RGB')
        garment_image = Image.open(garment_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        
        # Ensure mask is same size as model image
        if mask_image.size != model_image.size:
            mask_image = mask_image.resize(model_image.size, Image.LANCZOS)
        
        if is_reverse_task:
            # REVERSE TASK: Model wearing clothes -> Flat lay garment
            # Always crop the model for reverse task
            model_image, mask_image = self._apply_random_crop(model_image, mask_image)
            
            # Handle dropouts
            drop_prompt = random.random() < self.prompt_dropout_prob
            drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
            
            # Generate reverse instruction
            if drop_prompt:
                instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
            else:
                instruction_text = self._generate_instruction(folder_path, None, is_reverse=True)
                instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
            
            # Process input: cropped model image
            if not drop_ref_img:
                input_images = []
                input_images_path = [model_path]
                
                max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
                
                processed_model = self.image_processor.preprocess(
                    model_image, 
                    max_pixels=max_input_pixels, 
                    max_side_length=self.max_side_length
                )
                input_images.append(processed_model)
            else:
                input_images_path, input_images = None, None
            
            # Output: garment flat lay
            output_image = self.image_processor.preprocess(
                garment_image, 
                max_pixels=self.max_output_pixels, 
                max_side_length=self.max_side_length
            )
            
            data = {
                'task_type': 'reverse_virtual_try_on',
                'instruction': instruction,
                'input_images_path': input_images_path,
                'input_images': input_images,
                'output_image': output_image,
                'output_image_path': garment_path,
                'is_cropped': True,  # Always cropped for reverse task
                'mask_color': None,  # No color masking in reverse task
            }
            
        else:
            # FORWARD TASK: Original logic (Masked model + garment -> Model wearing clothes)
            # Decide whether to crop or not
            should_crop = random.random() < self.crop_prob
            
            if should_crop:
                # Apply random cropping to model image and mask
                model_image, mask_image = self._apply_random_crop(model_image, mask_image)
                # Note: garment image is not cropped as it should remain full
            
            # Select random color for this sample
            selected_color = self._select_random_color()
            
            # Create masked model image with selected color (after potential cropping)
            masked_model_image = self._create_masked_model_image(model_image, mask_image, selected_color)
            
            # Handle dropouts
            drop_prompt = random.random() < self.prompt_dropout_prob
            drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
            
            # Generate instruction with selected color
            if drop_prompt:
                instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
            else:
                instruction_text = self._generate_instruction(folder_path, selected_color, is_reverse=False)
                instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
            
            # Process input images
            if not drop_ref_img:
                input_images = []
                input_images_path = [model_path, garment_path]  # Store paths for reference
                
                max_input_pixels = self.max_input_pixels[1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
                
                # Process masked model image (potentially cropped)
                processed_masked_model = self.image_processor.preprocess(
                    masked_model_image, 
                    max_pixels=max_input_pixels, 
                    max_side_length=self.max_side_length
                )
                input_images.append(processed_masked_model)
                
                # Process garment image (always full image)
                processed_garment = self.image_processor.preprocess(
                    garment_image, 
                    max_pixels=max_input_pixels, 
                    max_side_length=self.max_side_length
                )
                input_images.append(processed_garment)
            else:
                input_images_path, input_images = None, None
            
            # Process output image (use cropped version if cropping was applied)
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
                'is_cropped': should_crop,  # Optional: track whether cropping was applied
                'mask_color': selected_color,  # Track which color was used
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