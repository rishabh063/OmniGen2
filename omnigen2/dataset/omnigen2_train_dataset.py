from typing import Optional, Union, List
import os
import json
import random
import torch
from PIL import Image
from torchvision import transforms as T
from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."
    CONSTANT_INSTRUCTION_CASE1 = "Fill the black area in image1 with clothing in image2"
    CONSTANT_INSTRUCTION_CASE2_TEMPLATE = "Put the garment on: {caption}"
    CONSTANT_INSTRUCTION_CASE3 = "Add detail and texture from image2 in image1"

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
        vton_data_dir="./vton",
        case_2_probability: float = 0,  # Probability of using case 2 (caption-based)
        case_3_probability: float = 0,  # Probability of using case 3 (enhancement)
        use_case_2_only: bool = False,    # If True, only use case 2
        use_case_1_only: bool = False,    # If True, only use case 1
        use_case_3_only: bool = False,    # If True, only use case 3
        img2img_strengths: List[str] = ["0_1", "0_2", "0_3"],  # Available strength suffixes
    ):
        self.vton_data_dir = vton_data_dir
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels
        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.condition_size = condition_size
        self.target_size = target_size
        self.case_2_probability = case_2_probability
        self.case_3_probability = case_3_probability
        self.use_case_2_only = use_case_2_only
        self.use_case_1_only = use_case_1_only
        self.use_case_3_only = use_case_3_only
        self.img2img_strengths = img2img_strengths

        # Validate case settings
        exclusive_cases = [use_case_1_only, use_case_2_only, use_case_3_only]
        if sum(exclusive_cases) > 1:
            raise ValueError("Cannot use multiple exclusive case options")

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)
        self.to_tensor = T.ToTensor()
        
        # Load local VTON dataset
        self.data_samples = self._load_vton_data()
        self.tokenizer = tokenizer
        
        # Filter samples based on case requirements
        self._filter_samples_by_case()
        
        print(f"VTON Dataset loaded with {len(self.data_samples)} samples")
        if not any([use_case_1_only, use_case_2_only, use_case_3_only]):
            print(f"Case 2 probability: {case_2_probability}")
            print(f"Case 3 probability: {case_3_probability}")
    
    def update_resolution_settings(self, max_output_pixels, max_input_pixels, max_side_length):
        """Update resolution settings during training"""
        self.max_output_pixels = max_output_pixels
        self.max_input_pixels = max_input_pixels
        self.max_side_length = max_side_length
        
        print(f"Dataset resolution updated: max_output_pixels={max_output_pixels}, "
              f"max_input_pixels={max_input_pixels}, max_side_length={max_side_length}")
    
    def _find_img2img_files(self, folder_path):
        """Find available img2img files with different strengths"""
        img2img_files = {
            'garment': [],
            'model': []
        }
        
        for filename in os.listdir(folder_path):
            if filename.startswith('garment_img2img_str') and filename.endswith('.jpg'):
                img2img_files['garment'].append(filename)
            elif filename.startswith('model_img2img_str') and filename.endswith('.jpg'):
                img2img_files['model'].append(filename)
        
        return img2img_files
    
    def _load_vton_data(self):
        """Load VTON data from local directory structure"""
        samples = []
        
        # Scan the vton directory for sample folders
        if not os.path.exists(self.vton_data_dir):
            raise FileNotFoundError(f"VTON data directory not found: {self.vton_data_dir}")
        
        for folder_name in os.listdir(self.vton_data_dir):
            folder_path = os.path.join(self.vton_data_dir, folder_name)
            
            # Skip files, only process directories
            if not os.path.isdir(folder_path):
                continue
            
            # Skip the processing summary file
            if folder_name in ["processing_summary.json", "caption_processing_log.json"]:
                continue
            
            # Check if this folder contains the required files
            required_files = ["model.jpg", "garment.jpg", "info.json"]
            case1_required = required_files + ["bboxed_model.jpg"]
            
            has_basic_files = all(os.path.exists(os.path.join(folder_path, f)) for f in required_files)
            has_case1_files = all(os.path.exists(os.path.join(folder_path, f)) for f in case1_required)
            
            # Check for img2img files for case 3
            img2img_files = self._find_img2img_files(folder_path)
            has_case3_files = len(img2img_files['garment']) > 0 and len(img2img_files['model']) > 0
            
            if has_basic_files:
                # Load metadata
                with open(os.path.join(folder_path, "info.json"), 'r') as f:
                    metadata = json.load(f)
                
                # Load captions if available
                captions = {}
                captions_path = os.path.join(folder_path, "captions.json")
                if os.path.exists(captions_path):
                    with open(captions_path, 'r') as f:
                        captions = json.load(f)
                elif 'captions' in metadata:
                    captions = metadata['captions']
                
                sample = {
                    'folder_path': folder_path,
                    'model_image_path': os.path.join(folder_path, "model.jpg"),
                    'garment_image_path': os.path.join(folder_path, "garment.jpg"),
                    'bboxed_image_path': os.path.join(folder_path, "bboxed_model.jpg") if has_case1_files else None,
                    'metadata': metadata,
                    'captions': captions,
                    'img2img_files': img2img_files,
                    'supports_case1': has_case1_files,
                    'supports_case2': bool(captions.get('model', {}).get('caption')),
                    'supports_case3': has_case3_files
                }
                samples.append(sample)
            else:
                print(f"Warning: Skipping incomplete sample folder: {folder_name}")
        
        print(f"Found {len(samples)} valid VTON samples")
        return samples
    
    def _filter_samples_by_case(self):
        """Filter samples based on case requirements"""
        if self.use_case_1_only:
            # Keep only samples that support case 1
            original_count = len(self.data_samples)
            self.data_samples = [s for s in self.data_samples if s['supports_case1']]
            print(f"Filtered to {len(self.data_samples)} samples supporting Case 1 (from {original_count})")
        
        elif self.use_case_2_only:
            # Keep only samples that support case 2
            original_count = len(self.data_samples)
            self.data_samples = [s for s in self.data_samples if s['supports_case2']]
            print(f"Filtered to {len(self.data_samples)} samples supporting Case 2 (from {original_count})")
            
        elif self.use_case_3_only:
            # Keep only samples that support case 3
            original_count = len(self.data_samples)
            self.data_samples = [s for s in self.data_samples if s['supports_case3']]
            print(f"Filtered to {len(self.data_samples)} samples supporting Case 3 (from {original_count})")
        
        else:
            # Count samples for each case
            case1_count = sum(1 for s in self.data_samples if s['supports_case1'])
            case2_count = sum(1 for s in self.data_samples if s['supports_case2'])
            case3_count = sum(1 for s in self.data_samples if s['supports_case3'])
            
            print(f"Samples supporting Case 1: {case1_count}")
            print(f"Samples supporting Case 2: {case2_count}")
            print(f"Samples supporting Case 3: {case3_count}")
    
    def apply_chat_template(self, instruction, system_prompt):
        """Apply chat template if enabled"""
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
    
    def load_and_process_image(self, image_path, max_pixels=None):
        """Load and process an image"""
        try:
            if image_path is None or not os.path.exists(image_path):
                return None
                
            image = Image.open(image_path).convert("RGB")
            
            # Process image using the image processor
            processed_image = self.image_processor.preprocess(
                image, 
                max_pixels=max_pixels, 
                max_side_length=self.max_side_length
            )
            
            return processed_image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def determine_case_for_sample(self, sample):
        """Determine which case to use for this sample"""
        if self.use_case_1_only:
            return 1 if sample['supports_case1'] else None
        elif self.use_case_2_only:
            return 2 if sample['supports_case2'] else None
        elif self.use_case_3_only:
            return 3 if sample['supports_case3'] else None
        else:
            # Multiple cases available, decide based on probability and support
            
            # Decide based on probabilities
            rand = random.random()
            
            
            # Select case based on normalized probabilities
            if rand < self.case_3_probability:
                return 3
            elif  rand < (self.case_3_probability + self.case_2_probability):
                return 2
            else :
                return 1
            # else:
            #     # Fallback to any available case
            #     return random.choice([1,2,3])
    
    def process_item_case1(self, sample):
        """Process sample using Case 1 (2 input images: bboxed + garment)"""
        # Handle prompt dropout
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(self.CONSTANT_INSTRUCTION_CASE1, self.SYSTEM_PROMPT)

        # Load images
        input_images = None
        input_images_path = None
        
        if not drop_ref_img:
            # Load bboxed image (image1) and garment image (image2)
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            
            bboxed_image = self.load_and_process_image(sample['bboxed_image_path'], max_input_pixels)
            garment_image = self.load_and_process_image(sample['garment_image_path'], max_input_pixels)
            
            if bboxed_image is not None and garment_image is not None:
                input_images = [bboxed_image, garment_image]
                input_images_path = ['bboxed_model', 'garment']
            else:
                return None

        # Load and process output image (original model image)
        output_image = self.load_and_process_image(sample['model_image_path'], self.max_output_pixels)
        
        if output_image is None:
            return None

        data = {
            'task_type': 'virtual_try_on_case1',
            'case_used': 1,
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': sample['model_image_path'],
            'metadata': sample['metadata']
        }
        return data
    
    def process_item_case2(self, sample):
        """Process sample using Case 2 (1 input image: garment + caption)"""
        # Get model caption for instruction
        model_caption = sample['captions'].get('model', {}).get('caption', '')
        ghost_mannequin_caption = sample['metadata'].get('type2', '')
        
        if not model_caption:
            return None
        
        # Handle prompt dropout
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        ghost_mannequin = False
        if random.random() < 0.5:
            ghost_mannequin = True
        
        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            if not ghost_mannequin:
                caption_instruction = self.CONSTANT_INSTRUCTION_CASE2_TEMPLATE.format(caption=model_caption)
                instruction = self.apply_chat_template(caption_instruction, self.SYSTEM_PROMPT)
            else:
                caption_instruction = f"Create a ghost mannequin of the {ghost_mannequin_caption}"
                instruction = self.apply_chat_template(caption_instruction, self.SYSTEM_PROMPT)
        
        # Load images
        input_images = None
        input_images_path = None
        
        if not drop_ref_img:
            # Load only garment image
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            if not ghost_mannequin:
                garment_image = self.load_and_process_image(sample['garment_image_path'], max_input_pixels)
                input_images_path = ['garment']
            else:
                garment_image = self.load_and_process_image(sample['model_image_path'], max_input_pixels)
                input_images_path = ['model']
            
            if garment_image is not None:
                input_images = [garment_image]
            else:
                return None
        
        outputimagepath = ''
        # Load and process output image (original model image)
        if not ghost_mannequin:
            output_image = self.load_and_process_image(sample['model_image_path'], self.max_output_pixels)
            outputimagepath = 'model'
        else:
            output_image = self.load_and_process_image(sample['garment_image_path'], self.max_output_pixels)
            outputimagepath = 'garment'
        
        if output_image is None:
            return None

        data = {
            'task_type': 'virtual_try_on_case2',
            'case_used': 2,
            'instruction': instruction,
            'model_caption': model_caption,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': sample['model_image_path'],
            'metadata': sample['metadata']
        }
        return data
    
    def process_item_case3(self, sample):
        """Process sample using Case 3 (enhancement: img2img + reference -> original)"""
        # Handle prompt dropout
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(self.CONSTANT_INSTRUCTION_CASE3, self.SYSTEM_PROMPT)

        # Randomly choose between garment enhancement or model enhancement
        enhance_garment = random.random() < 0.5
        
        # Load images
        input_images = None
        input_images_path = None
        
        if not drop_ref_img:
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            # print(max_input_pixels)
            if enhance_garment:
                # Enhance garment: garment_img2img + model -> garment
                if not sample['img2img_files']['garment']:
                    return None
                
                # Randomly select one of the available img2img garment files
                img2img_filename = random.choice(sample['img2img_files']['garment'])
                img2img_path = os.path.join(sample['folder_path'], img2img_filename)
                
                # Load img2img garment (image1) and original model (image2)
                img2img_image = self.load_and_process_image(img2img_path, max_input_pixels)
                reference_image = self.load_and_process_image(sample['model_image_path'], max_input_pixels)
                
                if img2img_image is not None and reference_image is not None:
                    input_images = [img2img_image, reference_image]
                    input_images_path = [img2img_filename, 'model']
                    output_image_path = sample['garment_image_path']
                    target_type = 'garment'
                else:
                    return None
            else:
                # Enhance model: model_img2img + garment -> model
                if not sample['img2img_files']['model']:
                    return None
                
                # Randomly select one of the available img2img model files
                img2img_filename = random.choice(sample['img2img_files']['model'])
                img2img_path = os.path.join(sample['folder_path'], img2img_filename)
                
                # Load img2img model (image1) and original garment (image2)
                img2img_image = self.load_and_process_image(img2img_path, max_input_pixels)
                reference_image = self.load_and_process_image(sample['garment_image_path'], max_input_pixels)
                
                if img2img_image is not None and reference_image is not None:
                    input_images = [img2img_image, reference_image]
                    input_images_path = [img2img_filename, 'garment']
                    output_image_path = sample['model_image_path']
                    target_type = 'model'
                else:
                    return None

        # Load and process output image (original target image)
        output_image = self.load_and_process_image(output_image_path, self.max_output_pixels)
        # print(self.max_output_pixels)
        if output_image is None:
            return None

        data = {
            'task_type': 'virtual_try_on_case3',
            'case_used': 3,
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'target_type': target_type,
            'metadata': sample['metadata']
        }
        return data

    def process_item(self, sample):
        """Process a single VTON sample based on determined case"""
        case_to_use = self.determine_case_for_sample(sample)
        
        if case_to_use is None:
            return None
        elif case_to_use == 1:
            return self.process_item_case1(sample)
        elif case_to_use == 2:
            return self.process_item_case2(sample)
        elif case_to_use == 3:
            return self.process_item_case3(sample)
        else:
            return None

    def __getitem__(self, index):
        """Get a single item from the dataset"""
        max_retries = 12

        current_index = index
        for attempt in range(max_retries):
            try:
                if current_index >= len(self.data_samples):
                    current_index = random.randint(0, len(self.data_samples) - 1)
                
                sample = self.data_samples[current_index]
                result = self.process_item(sample)
                
                if result is not None:
                    return result
                else:
                    # Try a different sample
                    current_index = random.randint(0, len(self.data_samples) - 1)
                    continue
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.data_samples) - 1)
                    continue
        
    def __len__(self):
        return len(self.data_samples)


class OmniGen2Collator():
    """Collator for VTON dataset - enhanced to support Case 3"""
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        # Filter out None items
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None
        
        task_type = [data['task_type'] for data in batch]
        case_used = [data.get('case_used', 1) for data in batch]
        instruction = [data['instruction'] for data in batch]
        input_images_path = [data['input_images_path'] for data in batch]
        input_images = [data['input_images'] for data in batch]
        output_image = [data['output_image'] for data in batch]
        output_image_path = [data['output_image_path'] for data in batch]
        metadata = [data.get('metadata', {}) for data in batch]
        model_caption = [data.get('model_caption', '') for data in batch]
        target_type = [data.get('target_type', '') for data in batch]

        text_inputs = self.tokenizer(
            instruction,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        data = {
            "task_type": task_type,
            "case_used": case_used,
            "text_ids": text_inputs.input_ids,
            "text_mask": text_inputs.attention_mask,
            "input_images": input_images, 
            "input_images_path": input_images_path,
            "output_image": output_image,
            "output_image_path": output_image_path,
            "metadata": metadata,
            "model_caption": model_caption,
            "target_type": target_type,
        }
        return data