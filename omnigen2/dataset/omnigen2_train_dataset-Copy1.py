from typing import Optional, Union, List
import os
import random
import yaml
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms as T
from datasets import load_dataset
from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

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
        condition_size: tuple = (1024, 1024),
        target_size: tuple = (1024, 1024),
        num_shards: int = 10,
    ):
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.condition_size = condition_size
        self.target_size = target_size

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.to_tensor = T.ToTensor()

        # Load HuggingFace dataset
        print(f"Loading {num_shards} shard(s) from jackyhate/text-to-image-2M...")
        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        urls = [base_url.format(i=i) for i in range(num_shards)]
        
        self.data = load_dataset("webdataset", data_files={"train": urls}, split="train")
        self.tokenizer = tokenizer
        print(f"Dataset loaded with {len(self.data)} samples")
    
    def __draw_regular_boxes__(self, mask, draw, w, h):
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
    
    def __draw_border_boxes__(self, mask, draw, w, h):
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
    
    def __draw_random_shapes__(self, mask, draw, w, h):
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
        
    def __get_condition__(self, image):
        """Generate masked condition image with three different strategies"""
        # condition_size = self.condition_size
        
        condition_img = image.convert("RGB")
        w, h = condition_img.size
        
        # Create mask
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Choose masking strategy
        strategy = random.choice(['regular_boxes', 'border_boxes', 'random_shapes'])
        
        if strategy == 'regular_boxes':
            # 30% of cases - regular boxes ensuring 30% visibility
            self.__draw_regular_boxes__(mask, draw, w, h)
        elif strategy == 'border_boxes':
            # Border-touching boxes
            self.__draw_border_boxes__(mask, draw, w, h)
        else:
            # Random shapes
            self.__draw_random_shapes__(mask, draw, w, h)
        
        # Randomly invert mask
        if random.random() > 0.05:
            mask = Image.eval(mask, lambda a: 255 - a)
        
        # Apply mask
        condition_img = Image.composite(
            condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        )
        
        return condition_img
    
    def clean_data_item(self, data_item):
        """Clean instruction text by removing common prefixes"""
        instruction = data_item['instruction']
        prefixes = ["The image portrays ", "The image depicts ", "The image captures ", 
                   "The image highlights ", "The image shows ", "这张图片展示了"]
        
        if random.random() < 0.5:
            for p in prefixes:
                if p in instruction:
                    instruction = instruction.replace(p, "")
                    break
        
        data_item['instruction'] = instruction
        return data_item
    
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
        # Extract image and prompt from webdataset format
        output_image = data_item['jpg']  # PIL Image
        # instruction = data_item['json']['prompt']  # Text prompt
        if random.random() > 0.5:
            instruction = "inpaint the black part , with appropriate image"
        else:
            instruction = data_item['json']['prompt']  # Text prompt
        # Resize output image to target size
        # output_image = output_image.resize(self.target_size).convert("RGB")
        output_image = output_image.convert("RGB")
        # Create data item in expected format
        processed_item = {
            'task_type': 'text_to_image',
            'instruction': instruction,
            'output_image': output_image
        }
        
        processed_item = self.clean_data_item(processed_item)

        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(processed_item['instruction'], self.SYSTEM_PROMPT)

        # Generate input condition image (masked version of output)
        input_images = None
        input_images_path = None
        
        if not drop_ref_img:
            condition_img = self.__get_condition__(output_image)
            
            # Process condition image same as input images
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            condition_img_processed = self.image_processor.preprocess(
                condition_img, 
                max_pixels=max_input_pixels, 
                max_side_length=self.max_side_length
            )
            input_images = [condition_img_processed]
            input_images_path = ['generated_condition']

        # Process output image
        output_image_processed = self.image_processor.preprocess(
            output_image, 
            max_pixels=self.max_output_pixels, 
            max_side_length=self.max_side_length
        )

        data = {
            'task_type': processed_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image_processed,
            'output_image_path': 'webdataset_image',
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