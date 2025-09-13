from typing import Optional, Union, List
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor
import torch
from torchvision import transforms as T
from datasets import load_dataset
from pathlib import Path
import logging



class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."
    
    INPAINTING_INSTRUCTION_TEMPLATES = [
        "inpaint the grey part with appropriate image",
        "Fill the grey masked area in the image",
        "Complete the grey missing parts of the image",
        "Inpaint the grey masked regions"
    ]
    
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
        face_dataset_dir: Optional[str] = "./human_Dataset/Original Images",
        background_removed_dataset_dir: Optional[str] = "./background_removed_dataset/dataset/progress/BiRefNet/downloads", 
        # HuggingFace inpainting dataset config
        hf_inpainting_dataset: str = "jackyhate/text-to-image-2M",
        hf_num_shards: int = 10,
    ):
        self.sampling_weights = {
            'hf': 0.5,
            'vton': 0.1,
            'face': 0.1,
            'background_removed': 0.3  # NEW
        }
        self.tokenizer = tokenizer
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        
        # Dataset directories
        self.vton_dataset_dir = vton_dataset_dir
        self.face_dataset_dir = face_dataset_dir
        self.background_removed_dataset_dir = background_removed_dataset_dir
        self.hf_inpainting_dataset = hf_inpainting_dataset
        self.hf_num_shards = hf_num_shards
        
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.use_chat_template = use_chat_template
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize all samples as inpainting samples
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize all datasets as inpainting sources"""
        self.hf_samples = []
        self.vton_samples = []
        self.face_samples = []
        self.background_removed_samples = []
        # Load HuggingFace dataset
        if self.hf_inpainting_dataset:
            try:
                from datasets import load_dataset
                
                self.logger.info(f"Loading {self.hf_num_shards} shard(s) from {self.hf_inpainting_dataset}...")
                base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
                urls = [base_url.format(i=i) for i in range(self.hf_num_shards)]
                
                self.hf_dataset = load_dataset("webdataset", data_files={"train": urls}, split="train")
                self.logger.info(f"HuggingFace dataset loaded with {len(self.hf_dataset)} samples")
                
                # Add HF samples
                for i in range(len(self.hf_dataset)):
                    self.hf_samples.append({
                        'type': 'hf_webdataset',
                        'index': i
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to load HuggingFace dataset: {e}")
        
        # Load VTON dataset images
        if self.vton_dataset_dir and os.path.exists(self.vton_dataset_dir):
            vton_count = 0
            for folder_name in os.listdir(self.vton_dataset_dir):
                folder_path = os.path.join(self.vton_dataset_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                
                model_path = os.path.join(folder_path, 'model.jpg')
                if os.path.exists(model_path):
                    self.vton_samples.append({
                        'type': 'vton',
                        'image_path': model_path,
                        'folder_path': folder_path
                    })
                    vton_count += 1
            
            self.logger.info(f"Found {vton_count} VTON images for inpainting")
        
        # Load face dataset images
        if self.face_dataset_dir and os.path.exists(self.face_dataset_dir):
            face_count = 0
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            for celebrity_folder in Path(self.face_dataset_dir).iterdir():
                if not celebrity_folder.is_dir():
                    continue
                
                for image_file in celebrity_folder.iterdir():
                    if (image_file.suffix.lower() in image_extensions and 
                        not image_file.name.endswith('_result.png')):
                        
                        self.face_samples.append({
                            'type': 'face',
                            'image_path': str(image_file),
                            'celebrity': celebrity_folder.name
                        })
                        face_count += 1
            
            self.logger.info(f"Found {face_count} face images for inpainting")
        
        if self.background_removed_dataset_dir and os.path.exists(self.background_removed_dataset_dir):
            bg_removed_count = 0
        
            for folder_name in os.listdir(self.background_removed_dataset_dir):
                folder_path = os.path.join(self.background_removed_dataset_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                
                base_image_path = os.path.join(folder_path, 'base.webp')
                if os.path.exists(base_image_path):
                    self.background_removed_samples.append({
                        'type': 'background_removed',
                        'image_path': base_image_path,
                        'folder_path': folder_path
                    })
                    bg_removed_count += 1
            
            self.logger.info(f"Found {bg_removed_count} background removed images for inpainting")
            # END NEW SECTION
    
        if len(self.hf_samples) + len(self.vton_samples) + len(self.face_samples) + len(self.background_removed_samples) == 0:  # UPDATED
            raise ValueError("No samples found in any dataset!")
    
        self.logger.info(f"Total samples available: {len(self.hf_samples) + len(self.vton_samples) + len(self.face_samples) + len(self.background_removed_samples)}")  # UPDATED
    
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
    
    def _draw_regular_boxes(self, mask, draw, w, h):
        """Draw regular boxes ensuring at least 30% of image is visible"""
        num_boxes = random.randint(1, 3)
        
        for _ in range(num_boxes):
            max_box_area = 0.7 * w * h
            
            box_w = random.randint(int(0.1 * w), int(0.6 * w))
            box_h = random.randint(int(0.1 * h), int(0.6 * h))
            
            if box_w * box_h > max_box_area:
                scale_factor = (max_box_area / (box_w * box_h)) ** 0.5
                box_w = int(box_w * scale_factor)
                box_h = int(box_h * scale_factor)
            
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
            touch_sides = random.sample(['top', 'bottom', 'left', 'right', 'none'], random.randint(1, 2))
            
            # Box dimensions (smaller for border boxes)
            box_w = random.randint(int(0.1 * w), int(0.4 * w))
            box_h = random.randint(int(0.1 * h), int(0.4 * h))
            
            # Position based on sides to touch
            if 'none' in touch_sides:
                margin = 10
                x1 = random.randint(margin, w - box_w - margin)
                y1 = random.randint(margin, h - box_h - margin)
            elif 'left' in touch_sides and 'top' in touch_sides:
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
                radius = random.randint(int(0.05 * min(w, h)), int(0.25 * min(w, h)))
                center_x = random.randint(radius, w - radius)
                center_y = random.randint(radius, h - radius)
                draw.ellipse([center_x - radius, center_y - radius, 
                            center_x + radius, center_y + radius], fill=255)
                
            elif shape_type == 'ellipse':
                width = random.randint(int(0.1 * w), int(0.3 * w))
                height = random.randint(int(0.1 * h), int(0.3 * h))
                x1 = random.randint(0, w - width)
                y1 = random.randint(0, h - height)
                draw.ellipse([x1, y1, x1 + width, y1 + height], fill=255)
                
            elif shape_type == 'polygon':
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
        """Generate procedural mask with different strategies"""
        w, h = image.size
        
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        strategy = random.choice(['regular_boxes', 'border_boxes', 'random_shapes'])
        
        if strategy == 'regular_boxes':
            self._draw_regular_boxes(mask, draw, w, h)
        elif strategy == 'border_boxes':
            self._draw_border_boxes(mask, draw, w, h)
        else:
            self._draw_random_shapes(mask, draw, w, h)
        
        if random.random() > 0.5:
            mask = Image.eval(mask, lambda a: 255 - a)
        
        return mask
    
    def _place_image_on_colored_canvas(self, image: Image.Image):
        """
        Creates a new image by placing the input image on a colored canvas.
        This is used to teach the model how to generate colorful backgrounds.
        Returns the new ground truth image and the mask for inpainting.
        """
        # 1. Create a canvas with a random size
        canvas_w = random.randint(512, 1024)
        canvas_h = random.randint(512, 1024)

        # 2. Color the canvas (50% white, 50% random non-black)
        if random.random() < 0.5:
            bg_color = (255, 255, 255)
        else:
            bg_color = (128, 128, 128)
            while bg_color == (128, 128, 128): # Ensure color is not black
                bg_color = tuple(random.randint(50, 255) for _ in range(3))
        
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

        # 3. Resize the input image to cover 25% to 90% of the canvas area
        canvas_area = canvas_w * canvas_h
        img_w, img_h = image.size
        target_area_ratio = random.uniform(0.25, 0.90)
        target_area = canvas_area * target_area_ratio

        # Calculate new dimensions while preserving aspect ratio
        img_aspect_ratio = img_w / img_h
        new_h = int((target_area / img_aspect_ratio) ** 0.5)
        new_w = int(img_aspect_ratio * new_h)

        # Ensure the resized image fits on the canvas
        if new_w > canvas_w or new_h > canvas_h:
            scale_factor = min(canvas_w / new_w, canvas_h / new_h)
            new_w = int(new_w * scale_factor)
            new_h = int(new_h * scale_factor)

        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 4. Randomly place this image on the new canvas
        paste_x = random.randint(0, canvas_w - new_w)
        paste_y = random.randint(0, canvas_h - new_h)
        
        # The new ground truth is the image pasted on the canvas
        ground_truth_image = canvas.copy()
        ground_truth_image.paste(resized_image, (paste_x, paste_y))

        

        return ground_truth_image

    def _create_masked_image(self, image, mask=None):
        """Create masked image for inpainting"""
        if mask is None:
            mask = self._generate_procedural_mask(image)
        
        # Apply mask to create masked image
        masked_image = Image.composite(
            image, Image.new("RGB", image.size, (128, 128, 128)), mask
        )
        
        return masked_image
    
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
    
    def process_sample(self, sample_data):
        """Process any sample as an inpainting task"""
        # Load image based on sample type
        if sample_data['type'] == 'hf_webdataset':
            hf_data = self.hf_dataset[sample_data['index']]
            image = hf_data['jpg'].convert('RGB')
            
            if random.random() > 0.75:
                instruction_text = random.choice(self.INPAINTING_INSTRUCTION_TEMPLATES)
            else:
                raw_instruction = hf_data['json']['prompt']
                instruction_text = self._clean_instruction_text(raw_instruction)
            
            image_path = 'webdataset_image'
            
        elif sample_data['type'] in ['vton', 'face' , 'background_removed']:
            image = Image.open(sample_data['image_path']).convert('RGB')
            instruction_text = random.choice(self.INPAINTING_INSTRUCTION_TEMPLATES)
            image_path = sample_data['image_path']
        
        # --- NEW: Color Generation Task ---
        precomputed_mask = None
        # if random.random() < 0.2:
        #     # For 20% of samples, create a color generation task
        #     image = self._place_image_on_colored_canvas(image)
        #     instruction_text = random.choice(self.INPAINTING_INSTRUCTION_TEMPLATES)
        #     # image_path is now synthetic, let's reflect that
        #     image_path = 'synthetic_color_generation'
        # --- END NEW ---

        # Create masked image with procedural mask (or the precomputed one)
        masked_image = self._create_masked_image(image, mask=precomputed_mask)
        
        # Handle prompt dropout
        drop_prompt = random.random() < self.prompt_dropout_prob
        
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(instruction_text, self.SYSTEM_PROMPT)
        
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        
        if not drop_ref_img:
            input_images = []
            input_images_path = [image_path]
            
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
            'output_image_path': image_path,
        }
    
    def __getitem__(self, index):
        max_retries = 12
        
        for attempt in range(max_retries):
            try:
                # First, select which dataset to sample from based on weights
                dataset_types = list(self.sampling_weights.keys())
                dataset_probs = list(self.sampling_weights.values())
                selected_dataset = random.choices(dataset_types, weights=dataset_probs, k=1)[0]
                
                # Then get a random sample from the selected dataset
                if selected_dataset == 'hf' and self.hf_samples:
                    sample_idx = random.randint(0, len(self.hf_samples) - 1)
                    sample_data = self.hf_samples[sample_idx]
                elif selected_dataset == 'vton' and self.vton_samples:
                    sample_idx = random.randint(0, len(self.vton_samples) - 1)
                    sample_data = self.vton_samples[sample_idx]
                elif selected_dataset == 'face' and self.face_samples:
                    sample_idx = random.randint(0, len(self.face_samples) - 1)
                    sample_data = self.face_samples[sample_idx]
                elif selected_dataset == 'background_removed' and self.background_removed_samples:  # NEW
                    
                    sample_idx = random.randint(0, len(self.background_removed_samples) - 1)
                    sample_data = self.background_removed_samples[sample_idx]
                    
                else:
                    all_samples = self.hf_samples + self.vton_samples + self.face_samples+self.background_removed_samples
                    # Handle case where one dataset might be empty
                    if not all_samples:
                        raise IndexError("No samples available to choose from.")
                    sample_idx = random.randint(0, len(all_samples) - 1)
                    sample_data = all_samples[sample_idx]
                
                return self.process_sample(sample_data)
                
            except Exception as e:
                self.logger.warning(f"Error processing sample (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise e
                else:
                    continue
    
    def __len__(self):
        # Return the size of the largest dataset to ensure all data can be seen
        return max(len(self.hf_samples), len(self.vton_samples), len(self.face_samples) +len(self.background_removed_samples), 1)


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