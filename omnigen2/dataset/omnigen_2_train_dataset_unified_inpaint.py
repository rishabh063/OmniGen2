from typing import Optional, Union, List
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor
import torch
from torchvision import transforms
from pathlib import Path
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)



class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."
    
    # Task-specific instruction templates
    VTON_INSTRUCTION_TEMPLATES = [
        "Fill the grey area in image1 with clothing in image2",
        "Fill the black area in image1 with clothing in image2", 
        "Fill the blue area in image1 with clothing in image2"
    ]
    
    INPAINTING_INSTRUCTION_TEMPLATES = [
        "Fill the masked area in the image",
        "Complete the missing parts of the image",
        "Restore the damaged areas in the image",
        "Inpaint the masked regions"
    ]
    
    FACE_INPAINTING_INSTRUCTION_TEMPLATES = [
        "Fill the black area in image1 using the reference face in image2",
        
    ]
    
    # Color mapping for VTON
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
        # Dataset paths
        vton_dataset_dir: Optional[str] = "./Simple_data3",
        inpainting_dataset_dir: Optional[str] = None,
        face_dataset_dir="./human_Dataset/Original Images",
        # HuggingFace inpainting dataset config
        hf_inpainting_dataset="jackyhate/text-to-image-2M",
        hf_num_shards: int = 1,  # Number of shards to load from HF dataset
        # Task sampling weights
        task_weights: Optional[List[float]] = None,  # [vton, inpainting, face_inpainting]
        # VTON parameters
        crop_prob: float = 0.7,
        min_padding: int = 25,
        max_padding: int = 100,
        min_crop_size: int = 256,
        color_weights: Optional[List[float]] = None,
        # Face inpainting parameters
        min_images_per_celebrity: int = 2,
        mask_coverage_range: tuple = (0.2, 0.8),
        augment_probability: float = 0.7,
        face_crop_padding: float = 0.3,
    ):
        self.tokenizer = tokenizer
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        
        # Dataset directories
        self.vton_dataset_dir = vton_dataset_dir
        self.inpainting_dataset_dir = inpainting_dataset_dir
        self.face_dataset_dir = face_dataset_dir
        self.hf_inpainting_dataset = hf_inpainting_dataset
        self.hf_num_shards = hf_num_shards
        
        # Task weights (default: equal probability for available tasks)
        available_tasks = []
        if vton_dataset_dir and os.path.exists(vton_dataset_dir):
            available_tasks.append('vton')
        # if inpainting_dataset_dir and os.path.exists(inpainting_dataset_dir):
        #     available_tasks.append('inpainting')
        if hf_inpainting_dataset:
            available_tasks.append('inpainting')
        if face_dataset_dir and os.path.exists(face_dataset_dir):
            for i in range(0,5):
                available_tasks.append('face_inpainting')
        
        self.available_tasks = available_tasks
        if task_weights is None:
            self.task_weights = [1.0] * len(available_tasks)
        else:
            self.task_weights = task_weights[:len(available_tasks)]
        
        # VTON parameters
        self.color_weights = color_weights if color_weights else [1.0, 1.0, 1.0]
        self.crop_prob = crop_prob
        self.min_padding = min_padding
        self.max_padding = max_padding
        self.min_crop_size = min_crop_size
        
        # Face inpainting parameters
        self.min_images_per_celebrity = min_images_per_celebrity
        self.mask_coverage_range = mask_coverage_range
        self.augment_probability = augment_probability
        self.face_crop_padding = face_crop_padding
        
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.use_chat_template = use_chat_template
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize datasets
        self._initialize_datasets()
        
        # Setup face augmentation transforms
        if 'face_inpainting' in self.available_tasks:
            self._setup_face_transforms()
    
    def _initialize_datasets(self):
        """Initialize all available datasets"""
        self.vton_samples = []
        self.inpainting_samples = []
        self.face_samples = []
        
        # Initialize VTON dataset
        if 'vton' in self.available_tasks:
            self.vton_samples = self._collect_vton_samples()
            self.logger.info(f"Found {len(self.vton_samples)} VTON samples")
        
        # Initialize inpainting dataset
        if 'inpainting' in self.available_tasks:
            self.inpainting_samples = self._collect_inpainting_samples()
            self.logger.info(f"Found {len(self.inpainting_samples)} inpainting samples")
        
        # Initialize face inpainting dataset
        if 'face_inpainting' in self.available_tasks:
            self.face_samples = self._collect_face_samples()
            self.logger.info(f"Found {len(self.face_samples)} face inpainting samples")
        
        # Calculate total dataset size
        self.total_samples = len(self.vton_samples) + len(self.inpainting_samples) + len(self.face_samples)
        if self.total_samples == 0:
            raise ValueError("No valid samples found in any dataset!")
    
    def _collect_vton_samples(self):
        """Collect VTON samples"""
        if not self.vton_dataset_dir or not os.path.exists(self.vton_dataset_dir):
            return []
        
        sample_folders = []
        for folder_name in os.listdir(self.vton_dataset_dir):
            folder_path = os.path.join(self.vton_dataset_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            model_path = os.path.join(folder_path, 'model.jpg')
            garment_path = os.path.join(folder_path, 'garment.jpg')
            mask_path = os.path.join(folder_path, 'mask.png')
            
            if all(os.path.exists(p) for p in [model_path, garment_path, mask_path]):
                sample_folders.append(folder_path)
        
        return sample_folders
    
    def _collect_inpainting_samples(self):
        """Collect general inpainting samples - support both HuggingFace webdataset and local directories"""
        samples = []
        
        # Load HuggingFace dataset if specified
        if self.hf_inpainting_dataset:
            try:
                from datasets import load_dataset
                
                self.logger.info(f"Loading {self.hf_num_shards} shard(s) from {self.hf_inpainting_dataset}...")
                
                if self.hf_inpainting_dataset == "jackyhate/text-to-image-2M":
                    # Special handling for the text-to-image-2M dataset
                    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
                    urls = [base_url.format(i=i) for i in range(self.hf_num_shards)]
                    
                    self.hf_dataset = load_dataset("webdataset", data_files={"train": urls}, split="train")
                    self.logger.info(f"HuggingFace dataset loaded with {len(self.hf_dataset)} samples")
                    
                    # Return a marker to indicate HF dataset is loaded
                    return ['hf_webdataset'] * len(self.hf_dataset)
                else:
                    # Generic HuggingFace dataset loading
                    self.hf_dataset = load_dataset(self.hf_inpainting_dataset, split="train")
                    self.logger.info(f"HuggingFace dataset {self.hf_inpainting_dataset} loaded with {len(self.hf_dataset)} samples")
                    
                    # Return indices for the HF dataset
                    return [f'hf_{i}' for i in range(len(self.hf_dataset))]
                    
            except Exception as e:
                self.logger.error(f"Failed to load HuggingFace dataset {self.hf_inpainting_dataset}: {e}")
                self.hf_dataset = None
        
        # Load local dataset if specified
        if self.inpainting_dataset_dir and os.path.exists(self.inpainting_dataset_dir):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            for root, dirs, files in os.walk(self.inpainting_dataset_dir):
                for file in files:
                    if file.lower().endswith(tuple(image_extensions)):
                        image_path = os.path.join(root, file)
                        # Look for corresponding mask (common naming conventions)
                        mask_candidates = [
                            os.path.join(root, file.rsplit('.', 1)[0] + '_mask.png'),
                            os.path.join(root, file.rsplit('.', 1)[0] + '_mask.jpg'),
                            os.path.join(root, 'mask_' + file),
                            os.path.join(root, file.replace('.', '_mask.')),
                        ]
                        
                        mask_path = None
                        for candidate in mask_candidates:
                            if os.path.exists(candidate):
                                mask_path = candidate
                                break
                        
                        # Add sample (with or without pre-existing mask)
                        samples.append({
                            'type': 'local',
                            'image_path': image_path,
                            'mask_path': mask_path,  # Can be None for generative masking
                            'has_mask': mask_path is not None
                        })
        
        return samples
    
    def _collect_face_samples(self):
        """Collect face inpainting samples"""
        if not self.face_dataset_dir or not os.path.exists(self.face_dataset_dir):
            return []
        
        celebrity_data = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for celebrity_folder in Path(self.face_dataset_dir).iterdir():
            if not celebrity_folder.is_dir():
                continue
                
            celebrity_name = celebrity_folder.name
            image_pairs = []
            
            for image_file in celebrity_folder.iterdir():
                if (image_file.suffix.lower() in image_extensions and 
                    not image_file.name.endswith('_result.png')):
                    
                    mask_file = celebrity_folder / f"{image_file.stem}_result.png"
                    if mask_file.exists():
                        image_pairs.append({
                            'image_path': str(image_file),
                            'mask_path': str(mask_file),
                            'celebrity': celebrity_name
                        })
            
            if len(image_pairs) >= self.min_images_per_celebrity:
                celebrity_data[celebrity_name] = image_pairs
        
        # Convert to flat list of potential pairs
        face_samples = []
        for celebrity, images in celebrity_data.items():
            face_samples.extend(images)
        
        return face_samples
    
    
    def _setup_face_transforms(self):
        """Setup face augmentation transforms using ReplayCompose"""
        self.face_augment_transforms = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        ], p=self.augment_probability)
            
                
        
    
    def _select_task(self):
        """Select which task to sample from"""
        return random.choices(self.available_tasks, weights=self.task_weights, k=1)[0]
    
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
    
    # VTON-specific methods (from original code)
    def _select_random_color(self):
        """Select a random color based on weights"""
        return random.choices(self.COLOR_NAMES, weights=self.color_weights, k=1)[0]
    
    def _get_mask_bbox(self, mask_image):
        """Get bounding box of the mask area"""
        mask_array = np.array(mask_image)
        coords = np.column_stack(np.where(mask_array > 0))
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (x_min, y_min, x_max, y_max)
    
    def _apply_random_crop(self, image, mask_image):
        """Apply random cropping around mask area with random padding"""
        bbox = self._get_mask_bbox(mask_image)
        
        if bbox is None:
            return image, mask_image
        
        x_min, y_min, x_max, y_max = bbox
        img_width, img_height = image.size
        
        padding = random.randint(self.min_padding, self.max_padding)
        
        crop_x_min = max(0, x_min - padding)
        crop_y_min = max(0, y_min - padding)
        crop_x_max = min(img_width, x_max + padding)
        crop_y_max = min(img_height, y_max + padding)
        
        crop_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        cropped_image = image.crop(crop_box)
        cropped_mask = mask_image.crop(crop_box)
        
        return cropped_image, cropped_mask
    
    def _create_masked_model_image(self, model_image, mask_image, color_name):
        """Create model image with masked areas filled with specified color"""
        model_array = np.array(model_image)
        mask_array = np.array(mask_image)
        
        mask_normalized = mask_array.astype(np.float32) / 255.0
        color_value = self.MASK_COLORS[color_name]
        
        if color_name == 'blue':
            color_image = np.full_like(model_array, color_value)
        else:
            color_image = np.full_like(model_array, color_value)
        
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        masked_model = model_array * (1 - mask_3d) + color_image * mask_3d
        
        return Image.fromarray(masked_model.astype(np.uint8))
    
    def _generate_vton_instruction(self, folder_path, color_name):
        """Generate VTON instruction"""
        folder_name = os.path.basename(folder_path).lower()
        
        if 'upper_body' in folder_name:
            category = 'upper body clothing'
        elif 'lower_body' in folder_name:
            category = 'lower body clothing'
        elif 'dresses' in folder_name:
            category = 'dress'
        else:
            category = 'clothing'
        
        color_index = self.COLOR_NAMES.index(color_name)
        template = self.VTON_INSTRUCTION_TEMPLATES[color_index]
        
        return template.format(category=category)
    
    # Face inpainting methods
    def _get_face_bounding_box(self, mask):
        """Extract bounding box from face segmentation mask"""
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask
            
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            h, w = mask_np.shape[:2]
            return [0, 0, w, h]
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h, w = mask_np.shape[:2]
        padding_x = int((x_max - x_min) * self.face_crop_padding)
        padding_y = int((y_max - y_min) * self.face_crop_padding)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return [x_min, y_min, x_max, y_max]
    
    def _crop_face_region(self, image, mask):
        """Crop face region from image using mask"""
        bbox = self._get_face_bounding_box(mask)
        x_min, y_min, x_max, y_max = bbox
        
        if isinstance(image, Image.Image):
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        else:
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
        return cropped_image, cropped_mask
    
    


    def _create_face_masked_image(self, image, mask):
        """Create face image with masking using three different approaches"""
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            mask_np = np.array(mask)
        else:
            img_np = image.copy()
            mask_np = mask.copy()
        
        # Find face pixels
        face_mask = mask_np > 0
        
        if not np.any(face_mask):
            return Image.fromarray(img_np) if isinstance(image, Image.Image) else img_np
        
        # Randomly select one of five masking approaches
        mask_type = random.randint(1, 10)
        # mask_type = 5
        if mask_type <3:
            # Full face masking - vectorized
            img_np[face_mask] = [0, 0, 0]
        
        elif mask_type <7:
            # Binary mask bbox with 0 margin
            # Find bounding box coordinates
            rows = np.any(face_mask, axis=1)
            cols = np.any(face_mask, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Apply bbox mask with no padding
            img_np[y_min:y_max+1, x_min:x_max+1] = [0, 0, 0]
        
        elif mask_type == 7:
            # Binary mask bbox with 20-50 margin
            padding = random.randint(20, 50)
            
            # Find bounding box coordinates
            rows = np.any(face_mask, axis=1)
            cols = np.any(face_mask, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Apply padding and ensure we don't go out of bounds
            height, width = img_np.shape[:2]
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            # Apply bbox mask with padding
            img_np[y_min:y_max+1, x_min:x_max+1] = [0, 0, 0]
        
        elif mask_type == 8:
            # Random partial bbox masking
            padding = random.randint(5, 20)
            
            # Find bounding box coordinates
            rows = np.any(face_mask, axis=1)
            cols = np.any(face_mask, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Apply padding to original bbox
            height, width = img_np.shape[:2]
            y_min_pad = max(0, y_min - padding)
            y_max_pad = min(height - 1, y_max + padding)
            x_min_pad = max(0, x_min - padding)
            x_max_pad = min(width - 1, x_max + padding)
            
            # Calculate midpoints
            y_mid = (y_min_pad + y_max_pad) // 2
            x_mid = (x_min_pad + x_max_pad) // 2
            
            # Randomly select which partial area to mask
            partial_options = [
                # Right half: from middle to right edge
                (y_min_pad, y_max_pad + 1, x_mid, x_max_pad + 1),
                # Left half: from left edge to middle
                (y_min_pad, y_max_pad + 1, x_min_pad, x_mid + 1),
                # Top half: from top edge to middle
                (y_min_pad, y_mid + 1, x_min_pad, x_max_pad + 1),
                # Bottom half: from middle to bottom edge
                (y_mid, y_max_pad + 1, x_min_pad, x_max_pad + 1),
                # Top-right quarter
                (y_min_pad, y_mid + 1, x_mid, x_max_pad + 1),
                # Top-left quarter
                (y_min_pad, y_mid + 1, x_min_pad, x_mid + 1),
                # Bottom-right quarter
                (y_mid, y_max_pad + 1, x_mid, x_max_pad + 1),
                # Bottom-left quarter
                (y_mid, y_max_pad + 1, x_min_pad, x_mid + 1)
            ]
            
            # Select random partial area
            y_start, y_end, x_start, x_end = random.choice(partial_options)
            
            # Apply partial bbox mask
            img_np[y_start:y_end, x_start:x_end] = [0, 0, 0]
        
        else:
            # Random continuous patches of the mask
            # Get all face pixel coordinates
            face_coords = np.argwhere(face_mask)
            
            if len(face_coords) == 0:
                return Image.fromarray(img_np) if isinstance(image, Image.Image) else img_np
            
            # Parameters for patch generation
            num_patches = random.randint(1, 3)  # 1-3 patches
            coverage = random.uniform(0.3, 0.7)  # 30-70% of face area
            target_pixels = int(len(face_coords) * coverage)
            
            # Create a mask for selected continuous patches
            patch_mask = np.zeros_like(face_mask, dtype=bool)
            pixels_masked = 0
            
            for _ in range(num_patches):
                if pixels_masked >= target_pixels:
                    break
                
                # Select a random starting point within the face
                start_idx = random.randint(0, len(face_coords) - 1)
                start_y, start_x = face_coords[start_idx]
                
                # Generate patch size (square patches)
                remaining_pixels = target_pixels - pixels_masked
                max_patch_size = int(np.sqrt(remaining_pixels / num_patches)) + random.randint(5, 15)
                patch_size = random.randint(10, max_patch_size)
                
                # Define patch boundaries
                y_start = max(0, start_y - patch_size // 2)
                y_end = min(face_mask.shape[0], start_y + patch_size // 2)
                x_start = max(0, start_x - patch_size // 2)
                x_end = min(face_mask.shape[1], start_x + patch_size // 2)
                
                # Only mask pixels that are within the original face mask
                patch_region = face_mask[y_start:y_end, x_start:x_end]
                patch_mask[y_start:y_end, x_start:x_end] |= patch_region
                
                # Count newly masked pixels
                pixels_masked = np.sum(patch_mask)
            
            # Apply the continuous patch mask
            img_np[patch_mask] = [0, 0, 0]
        
        return Image.fromarray(img_np) if isinstance(image, Image.Image) else img_np    
    
    def _draw_regular_boxes(self, mask, draw, w, h):
        """Draw regular boxes ensuring at least 30% of image is visible"""
        num_boxes = random.randint(1, 3)
        
        for _ in range(num_boxes):
            # Ensure box doesn't cover more than 70% of the image
            max_box_area = 0.7 * w * h
            
            # Generate box dimensions
            box_w = random.randint(int(0.1 * w), int(0.6 * w))
            box_h = random.randint(int(0.1 * h), int(0.6 * h))
            
            # Ensure area constraint
            if box_w * box_h > max_box_area:
                scale_factor = (max_box_area / (box_w * box_h)) ** 0.5
                box_w = int(box_w * scale_factor)
                box_h = int(box_h * scale_factor)
            
            # Random position
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            draw.rectangle([x1, y1, x2, y2], fill=255)
    
    def _draw_border_boxes(self, mask, draw, w, h):
        """Draw boxes touching image borders"""
        num_boxes = random.randint(1, 4)
        
        for _ in range(num_boxes):
            # Choose which side(s) to touch
            touch_sides = random.sample(['top', 'bottom', 'left', 'right'], random.randint(1, 2))
            
            # Box dimensions (smaller for border boxes)
            box_w = random.randint(int(0.1 * w), int(0.4 * w))
            box_h = random.randint(int(0.1 * h), int(0.4 * h))
            
            # Position based on sides to touch
            if 'left' in touch_sides and 'top' in touch_sides:
                x1, y1 = 0, 0
            elif 'right' in touch_sides and 'top' in touch_sides:
                x1, y1 = w - box_w, 0
            elif 'left' in touch_sides and 'bottom' in touch_sides:
                x1, y1 = 0, h - box_h
            elif 'right' in touch_sides and 'bottom' in touch_sides:
                x1, y1 = w - box_w, h - box_h
            elif 'left' in touch_sides:
                x1 = 0
                y1 = random.randint(0, h - box_h)
            elif 'right' in touch_sides:
                x1 = w - box_w
                y1 = random.randint(0, h - box_h)
            elif 'top' in touch_sides:
                x1 = random.randint(0, w - box_w)
                y1 = 0
            elif 'bottom' in touch_sides:
                x1 = random.randint(0, w - box_w)
                y1 = h - box_h
            
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            draw.rectangle([x1, y1, x2, y2], fill=255)
    
    def _draw_random_shapes(self, mask, draw, w, h):
        """Draw random shapes like circles, ellipses, polygons"""
        num_shapes = random.randint(1, 3)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'ellipse', 'polygon'])
            
            if shape_type == 'circle':
                # Circle
                radius = random.randint(int(0.05 * min(w, h)), int(0.25 * min(w, h)))
                center_x = random.randint(radius, w - radius)
                center_y = random.randint(radius, h - radius)
                draw.ellipse([center_x - radius, center_y - radius, 
                            center_x + radius, center_y + radius], fill=255)
                
            elif shape_type == 'ellipse':
                # Ellipse
                width = random.randint(int(0.1 * w), int(0.3 * w))
                height = random.randint(int(0.1 * h), int(0.3 * h))
                x1 = random.randint(0, w - width)
                y1 = random.randint(0, h - height)
                draw.ellipse([x1, y1, x1 + width, y1 + height], fill=255)
                
            elif shape_type == 'polygon':
                # Random polygon (triangle to hexagon)
                num_points = random.randint(3, 6)
                center_x = random.randint(int(0.2 * w), int(0.8 * w))
                center_y = random.randint(int(0.2 * h), int(0.8 * h))
                max_radius = min(center_x, center_y, w - center_x, h - center_y)
                radius = random.randint(int(0.1 * max_radius), int(0.8 * max_radius))
                
                points = []
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points + random.uniform(-0.5, 0.5)
                    point_radius = radius * random.uniform(0.7, 1.3)
                    x = center_x + point_radius * np.cos(angle)
                    y = center_y + point_radius * np.sin(angle)
                    points.append((int(x), int(y)))
                
                draw.polygon(points, fill=255)
    
    def _generate_procedural_mask(self, image):
        """Generate procedural mask with three different strategies"""
        condition_img = image.convert("RGB")
        w, h = condition_img.size
        
        # Create mask
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Choose masking strategy
        strategy = random.choice(['regular_boxes', 'border_boxes', 'random_shapes'])
        
        if strategy == 'regular_boxes':
            self._draw_regular_boxes(mask, draw, w, h)
        elif strategy == 'border_boxes':
            self._draw_border_boxes(mask, draw, w, h)
        else:
            self._draw_random_shapes(mask, draw, w, h)
        
        # Randomly invert mask (95% chance to keep as is)
        if random.random() > 0.95:
            mask = Image.eval(mask, lambda a: 255 - a)
        
        return mask
    
    def _create_general_masked_image(self, image, mask=None):
        """Create general inpainting masked image - either use provided mask or generate procedural mask"""
        if mask is None:
            # Generate procedural mask
            mask = self._generate_procedural_mask(image)
        
        # Apply mask to create masked image
        condition_img = Image.composite(
            image, Image.new("RGB", image.size, (0, 0, 0)), mask
        )
        
        return condition_img
    
    def _clean_instruction_text(self, instruction):
        """Clean instruction text by removing common prefixes"""
        prefixes = ["The image portrays ", "The image depicts ", "The image captures ", 
                   "The image highlights ", "The image shows ", "这张图片展示了"]
        
        if random.random() < 0.5:
            for p in prefixes:
                if p in instruction:
                    instruction = instruction.replace(p, "")
                    break
        
        return instruction
    
    def process_vton_item(self, folder_path):
        """Process VTON sample"""
        model_path = os.path.join(folder_path, 'model.jpg')
        if random.random() > 0.95:
            garment_path = os.path.join(folder_path, 'garment.jpg')
        else:
            garment_path = os.path.join(folder_path, 'clipped_garment.png')
        mask_path = os.path.join(folder_path, 'mask.png')
        
        model_image = Image.open(model_path).convert('RGB')
        garment_image = Image.open(garment_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        
        if mask_image.size != model_image.size:
            mask_image = mask_image.resize(model_image.size, Image.LANCZOS)
        
        should_crop = random.random() < self.crop_prob
        if should_crop:
            model_image, mask_image = self._apply_random_crop(model_image, mask_image)
        
        selected_color = self._select_random_color()
        masked_model_image = self._create_masked_model_image(model_image, mask_image, selected_color)
        
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction_text = self._generate_vton_instruction(folder_path, selected_color)
            instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
        if not drop_ref_img:
            input_images = []
            input_images_path = [model_path, garment_path]
            
            max_input_pixels = self.max_input_pixels[1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
            processed_masked_model = self.image_processor.preprocess(
                masked_model_image, 
                max_pixels=max_input_pixels, 
                max_side_length=self.max_side_length
            )
            input_images.append(processed_masked_model)
            
            processed_garment = self.image_processor.preprocess(
                garment_image, 
                max_pixels=max_input_pixels, 
                max_side_length=self.max_side_length
            )
            input_images.append(processed_garment)
        else:
            input_images_path, input_images = None, None
        
        output_image = self.image_processor.preprocess(
            model_image, 
            max_pixels=self.max_output_pixels, 
            max_side_length=self.max_side_length
        )
        
        return {
            'task_type': 'virtual_try_on',
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': model_path,
        }
    
    def process_inpainting_item(self, sample_data):
        """Process general inpainting sample with sophisticated masking"""
        # Handle HuggingFace webdataset format
        if isinstance(sample_data, dict) and 'jpg' in sample_data:
            # This is from HuggingFace webdataset
            image = sample_data['jpg'].convert('RGB')
            mask = None  # Will generate procedural mask
            
            # Get instruction
            if random.random() > 0.5:
                instruction_text = "inpaint the black part, with appropriate image"
            else:
                raw_instruction = sample_data['json']['prompt']
                instruction_text = self._clean_instruction_text(raw_instruction)
        else:
            # This is from local dataset
            image = Image.open(sample_data['image_path']).convert('RGB')
            
            # Load mask if available, otherwise generate procedural mask
            if sample_data.get('has_mask', False) and sample_data['mask_path']:
                mask = Image.open(sample_data['mask_path']).convert('L')
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.LANCZOS)
            else:
                mask = None  # Will generate procedural mask
            
            # Get instruction - for local datasets, try to get from JSON if available
            if random.random() > 0.5:
                instruction_text = "inpaint the black part, with appropriate image"
            else:
                # Try to get prompt from JSON if available, otherwise use default
                if 'json' in sample_data and 'prompt' in sample_data['json']:
                    instruction_text = self._clean_instruction_text(sample_data['json']['prompt'])
                else:
                    instruction_text = random.choice(self.INPAINTING_INSTRUCTION_TEMPLATES)
        
        # Create masked image (either using provided mask or generating procedural mask)
        masked_image = self._create_general_masked_image(image, mask)
        
        drop_prompt = random.random() < self.prompt_dropout_prob
        
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
        if not drop_ref_img:
            input_images = []
            if isinstance(sample_data, dict) and 'jpg' in sample_data:
                input_images_path = ['webdataset_image']
            else:
                input_images_path = [sample_data['image_path']]
            
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
            processed_masked = self.image_processor.preprocess(
                masked_image,
                max_pixels=max_input_pixels,
                max_side_length=self.max_side_length
            )
            input_images.append(processed_masked)
        else:
            input_images_path, input_images = None, None
        
        output_image = self.image_processor.preprocess(
            image,
            max_pixels=self.max_output_pixels,
            max_side_length=self.max_side_length
        )
        
        return {
            'task_type': 'inpainting',
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': sample_data.get('image_path', 'webdataset_image'),
        }
    def apply_synchronized_augmentations(self, image1, image2):
        """Apply the same augmentation transforms to two images synchronously."""
        # Apply transforms to first image and get replay data
        return image1,image2
        augmented_data1 = self.face_augment_transforms(image=np.array(image1))
        augmented_img1 = Image.fromarray(augmented_data1['image'])
        
        # Apply the same transforms to second image using replay
        augmented_data2 = A.ReplayCompose.replay(augmented_data1['replay'], image=np.array(image2))
        augmented_img2 = Image.fromarray(augmented_data2['image'])
            
        return augmented_img1, augmented_img2

    def process_face_item(self, sample_data):
        """Process face inpainting sample"""
        # Get random reference image from same celebrity
        celebrity = sample_data['celebrity']
        celebrity_images = [img for img in self.face_samples if img['celebrity'] == celebrity]
        
        if len(celebrity_images) < 2:
            # Fallback to any other celebrity if needed
            celebrity_images = [img for img in self.face_samples if img['celebrity'] != celebrity]
        
        reference_data = random.choice([img for img in celebrity_images if img != sample_data])
        
        # Load images
        target_image = Image.open(sample_data['image_path']).convert('RGB')
        target_mask = Image.open(sample_data['mask_path']).convert('L')
        reference_image = Image.open(reference_data['image_path']).convert('RGB')
        reference_mask = Image.open(reference_data['mask_path']).convert('L')
        
        # Crop face regions
        target_face, target_face_mask = self._crop_face_region(target_image, target_mask)
        reference_face, _ = self._crop_face_region(reference_image, reference_mask)
        
        # Apply augmentations
        if random.random() < self.augment_probability:
            target_face, target_face_mask = self.apply_synchronized_augmentations(
                target_face, target_face_mask
            )
        # if random.random() < self.augment_probability:
        #     augmented_ref = self.face_augment_transforms(image=np.array(reference_face))['image']
        #     reference_face = Image.fromarray(augmented_ref)
        
        
        # Create masked version
        masked_face = self._create_face_masked_image(target_face, target_face_mask)
        
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction_text = random.choice(self.FACE_INPAINTING_INSTRUCTION_TEMPLATES)
            instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
        if not drop_ref_img:
            input_images = []
            input_images_path = [reference_data['image_path'], sample_data['image_path']]
            
            max_input_pixels = self.max_input_pixels[1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
            

            # Process masked face
            processed_masked = self.image_processor.preprocess(
                masked_face,
                max_pixels=max_input_pixels,
                max_side_length=self.max_side_length
            )
            input_images.append(processed_masked)

            # Process reference face
            processed_reference = self.image_processor.preprocess(
                reference_face,
                max_pixels=max_input_pixels,
                max_side_length=self.max_side_length
            )
            input_images.append(processed_reference)
            
            
        else:
            input_images_path, input_images = None, None
        
        output_image = self.image_processor.preprocess(
            target_face,
            max_pixels=self.max_output_pixels,
            max_side_length=self.max_side_length
        )
        
        return {
            'task_type': 'face_inpainting',
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': sample_data['image_path'],
        }
    
    def __getitem__(self, index):
            max_retries = 12
            timeout_seconds = 1  # 30 second timeout per sample
            
            for attempt in range(max_retries):
                try:
                    with timeout(timeout_seconds):
                        # Select task
                        task = self._select_task()
                        # print(f"Attempting task: {task} (attempt {attempt + 1})")
                        
                        if task == 'vton' and self.vton_samples:
                            sample_idx = random.randint(0, len(self.vton_samples) - 1)
                            folder_path = self.vton_samples[sample_idx]
                            element_to_return=self.process_vton_item(folder_path)
                            
                            return element_to_return
                        
                        elif task == 'inpainting' and self.inpainting_samples:
                            sample_idx = random.randint(0, len(self.inpainting_samples) - 1)
                            sample_identifier = self.inpainting_samples[sample_idx]
                            
                            # Check if this is a HuggingFace dataset sample
                            if isinstance(sample_identifier, str) and sample_identifier.startswith('hf_'):
                                if sample_identifier == 'hf_webdataset':
                                    # Get random sample from webdataset
                                    hf_idx = random.randint(0, len(self.hf_dataset) - 1)
                                    sample_data = self.hf_dataset[hf_idx]
                                else:
                                    # Extract index for regular HF dataset
                                    hf_idx = int(sample_identifier.split('_')[1])
                                    sample_data = self.hf_dataset[hf_idx]
                                
                                return self.process_inpainting_item(sample_data)
                            else:
                                # Local dataset sample
                                sample_data = sample_identifier
                                return self.process_inpainting_item(sample_data)
                        
                        elif task == 'face_inpainting' and self.face_samples:
                            sample_idx = random.randint(0, len(self.face_samples) - 1)
                            sample_data = self.face_samples[sample_idx]
                            
                            return self.process_face_item(sample_data)
                        
                        else:
                            # Fallback to any available task if selected task has no samples
                            available_tasks_with_data = []
                            if self.vton_samples:
                                available_tasks_with_data.append('vton')
                            if self.inpainting_samples:
                                available_tasks_with_data.append('inpainting')
                            if self.face_samples:
                                available_tasks_with_data.append('face_inpainting')
                            
                            if not available_tasks_with_data:
                                raise ValueError("No samples available in any dataset!")
                            
                            task = random.choice(available_tasks_with_data)
                            continue
                            
                except TimeoutError as e:
                    print(f"Timeout on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to load sample after {max_retries} attempts due to timeouts")
                    else:
                        continue
                        
                except Exception as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    self.logger.warning(f"Error processing sample (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    else:
                        continue    
    def __len__(self):
        return max(1, self.total_samples)


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


