import numpy as np
if not hasattr(np, 'int'):
    np.int = int


import time
import torch
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import dotenv
from PIL import Image, ImageOps
from typing import List, Tuple
dotenv.load_dotenv(override=True)

import time
from copy import deepcopy
import argparse
import logging
import math
import os
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm



import matplotlib.pyplot as plt



import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision.transforms.functional import crop, to_pil_image, to_tensor

from einops import repeat, rearrange

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLModel as TextEncoder

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from peft import LoraConfig

from omnigen2.training_utils import EMAModel
from omnigen2.utils.logging_utils import TqdmToLogger
from omnigen2.transport import create_transport
from omnigen2.dataset.inpaintPro import OmniGen2TrainDataset, OmniGen2Collator
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from transformers import Qwen2_5_VLProcessor
from transformers import Qwen2TokenizerFast
logger = get_logger(__name__)



def create_dataloader_for_stage(stage_config, args, text_tokenizer, worker_init_fn):
    """Create a new dataloader for the current training stage"""
    
    # Create new dataset with stage-specific parameters
    train_dataset = OmniGen2TrainDataset(
        tokenizer=text_tokenizer,
        use_chat_template=args.data.use_chat_template,
        prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
        ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
        max_input_pixels=stage_config['max_pixels'],  # Use stage resolution
        max_output_pixels=stage_config['max_pixels'], # Use stage resolution
        max_side_length=stage_config['resolution'],   # Use stage resolution
    )
    
    # Create new dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        collate_fn=OmniGen2Collator(tokenizer=text_tokenizer, max_token_len=args.data.maximum_text_tokens)
    )
    
    return train_dataset, train_dataloader




def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=2,
        help="Number of webdataset shards to load from jackyhate/text-to-image-2M",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    # output_dir = os.path.join(root_path, 'experiments', conf.name)+str(time.time())
    output_dir = os.path.join(root_path, 'experiments', "ft_lora_proper_inpaint_bg_dataset_no_margin")
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    # Add num_shards to config
    conf.data.num_shards = args.num_shards
    return conf

def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)


def log_model_info(name: str, model: torch.nn.Module):
    """Logs parameter counts for a given model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")


def log_time_distribution(transport, device, args):
    """Samples time steps from transport and plots their distribution."""
    with torch.no_grad():
        dummy_tensor = torch.randn((64, 16, int(math.sqrt(args.data.max_output_pixels) / 8), int(math.sqrt(args.data.max_output_pixels) / 8)), device=device)
        ts = torch.cat([transport.sample(dummy_tensor, AcceleratorState().process_index, AcceleratorState().num_processes)[0] for _ in range(1000)], dim=0)
    
    ts_np = ts.cpu().numpy()
    percentile_70 = np.percentile(ts_np, 70)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ts_np, bins=50, edgecolor='black', alpha=0.7, label="Time Step Distribution")
    plt.axvline(percentile_70, color='red', linestyle='dashed', linewidth=2, label=f'70th Percentile = {percentile_70:.2f}')
    plt.title('Distribution of Sampled Time Steps (t)')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = Path(args.output_dir) / 't_distribution.png'
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Time step distribution plot saved to {save_path}")



import numpy as np
from PIL import Image

def place_image_on_canvas(img, canvas_width, canvas_height, alignment='center', background_color=(128,128,128)):
    """
    Place an image on a canvas with specified dimensions and alignment.
    
    Args:
        img: PIL Image - input image to place on canvas
        canvas_width: int - width of the canvas
        canvas_height: int - height of the canvas
        alignment: str - alignment option ('top', 'bottom', 'left', 'right', 'center', 
                        'top-left', 'top-right', 'bottom-left', 'bottom-right', 'bottom-center')
        background_color: tuple - RGB color for canvas background (default: white)
        
    Returns:
        PIL Image - image placed on canvas with specified alignment
    """
    # Create canvas with specified dimensions and background color
    if len(background_color) == 3:
        canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
    else:
        canvas = Image.new('RGBA', (canvas_width, canvas_height), background_color)
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Calculate position based on alignment
    if alignment == 'center':
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
    elif alignment == 'top':
        x = (canvas_width - img_width) // 2
        y = 0
    elif alignment == 'bottom':
        x = (canvas_width - img_width) // 2
        y = canvas_height - img_height
    elif alignment == 'bottom-center':
        x = (canvas_width - img_width) // 2
        y = canvas_height - img_height
    elif alignment == 'left':
        x = 0
        y = (canvas_height - img_height) // 2
    elif alignment == 'right':
        x = canvas_width - img_width
        y = (canvas_height - img_height) // 2
    elif alignment == 'top-left':
        x = 0
        y = 0
    elif alignment == 'top-right':
        x = canvas_width - img_width
        y = 0
    elif alignment == 'bottom-left':
        x = 0
        y = canvas_height - img_height
    elif alignment == 'bottom-right':
        x = canvas_width - img_width
        y = canvas_height - img_height
    else:
        # Default to center if alignment not recognized
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
    
    # Ensure coordinates are within bounds
    x = max(0, min(x, canvas_width - img_width))
    y = max(0, min(y, canvas_height - img_height))
    
    # Paste the image onto the canvas
    if img.mode == 'RGBA':
        canvas.paste(img, (x, y), img)  # Use alpha channel for transparency
    else:
        canvas.paste(img, (x, y))
    
    return canvas






def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, 'logs'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=OmegaConf.to_object(args.logger.log_with),
        project_config=accelerator_project_config,
    )



    
    qwen_processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    def preprocess_fixed_images(input_image_path: List[str] = []) -> List[Image.Image]:
        """Preprocess the input images."""
        from PIL import Image, ImageOps
        import os
        
        input_images = []
        if input_image_path:
            if isinstance(input_image_path, str):
                input_image_path = [input_image_path]
            if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
                input_images = [Image.open(os.path.join(input_image_path[0], f)).convert('RGB') 
                              for f in os.listdir(input_image_path[0])]
            else:
                input_images = [Image.open(path).convert('RGB') for path in input_image_path]
            input_images = [ImageOps.exif_transpose(img) for img in input_images]
        return input_images
    
    def resize_image(image, max_size=1024):
        """Resize image while maintaining aspect ratio"""
        from PIL import Image
        
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
    
    def get_fixed_test_cases(root_dir):
        """Get fixed test cases for evaluation during training"""
        negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        
        # Define your fixed test cases

        inputs = [
#             (f""""<|im_start|>system
# You are a helpful assistant that generates high-quality virtual try-on images based on user instructions.<|im_end|>
# <|im_start|>user
# Fill the black area in image1 with clothing in image2<|im_end|>
# """ , 
#              ["vton/262208_front/bboxed_model.jpg",
#               "example_images/bf5e132d9b04e347365118427592e33e39ae5bad.jpg"]),
#             (f""""<|im_start|>system
# You are a helpful assistant that generates high-quality virtual try-on images based on user instructions.<|im_end|>
# <|im_start|>user
# Fill the black area in image1 with clothing in image2<|im_end|>
# """ , 
#              ["vton/262208_front/bboxed_model.jpg",
#               "example_images/f3979db3a2dab4a04259b0bef35ae5719bb52bc8.jpg"]),
#             (f""""<|im_start|>system
# You are a helpful assistant that generates high-quality virtual try-on images based on user instructions.<|im_end|>
# <|im_start|>user
# Fill the black area in image1 with clothing in image2<|im_end|>
# """ , 
#              ["vton/262208_front/bboxed_model.jpg",
#               "example_images/4442fbac4e3080ec20b2f14e353fea267249b0dd.jpg"]),

#             ( "Fill the masked face area in image2 using the reference face in image1",[ "human_test/inpaintimg.jpg", "human_Dataset/Original Images/Tom Cruise/Tom Cruise_0.jpg"]),
#             ( "Fill the masked face area in image2 using the reference face in image1",[ "human_test/inpaintimg.jpg", "human_Dataset/Original Images/Hrithik Roshan/Hrithik Roshan_6.jpg"]),
#             ( "Fill the masked face area in image2 using the reference face in image1",[ "human_test/inpaintimg.jpg", "human_Dataset/Original Images/Amitabh Bachchan/Amitabh Bachchan_29.jpg"]),

            ("inpaint the grey part with appropriate image" , ["vton/262208_front/bboxed_model.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/bf5e132d9b04e347365118427592e33e39ae5bad.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/f3979db3a2dab4a04259b0bef35ae5719bb52bc8.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/4442fbac4e3080ec20b2f14e353fea267249b0dd.jpg"]),
            ("inpaint the grey part with appropriate image" , ["human_Dataset/Original Images/Tom Cruise/Tom Cruise_0.jpg"]),
            ("inpaint the grey part with appropriate image" , ["human_Dataset/Original Images/Hrithik Roshan/Hrithik Roshan_6.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/bottom_1757176369_8.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/bottom_1757176525_5.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/back_1757176533_3.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/left_1757180555_3.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/top_1757180555_5.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/back_1757180564_4.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/bottom_1757180564_5.jpg"]),
            ("inpaint the grey part with appropriate image" , ["example_images/realimages/top_1757181631_6.jpg"]),
             
        ]
        
        return inputs, negative_prompt
    
    def generate_full_images_during_training(
        pipeline, 
        global_step, 
        args, 
        num_inference_steps=50,
        text_guidance_scale=4.0,
        image_guidance_scale=2.0
    ):
        """Generate full denoised images during training for evaluation using fixed test cases"""
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Create the pipeline from your current models
        pipeline.transformer = accelerator.unwrap_model(model)
        if ema_decay != 0:
            # Use EMA model for better quality
            pipeline.transformer = accelerator.unwrap_model(model_ema.averaged_model)
        
        pipeline.transformer.eval()
        
        # Get fixed test cases
        inputs, negative_prompt = get_fixed_test_cases(args.root_dir)
        
        with torch.no_grad():
            for test_idx, (instruction, input_image_paths) in enumerate(inputs):
                
                    # Preprocess input images
                input_images = preprocess_fixed_images(input_image_paths)
                
                    
                    # Resize images if needed
                for i in range(len(input_images)):
                    input_images[i] = resize_image(place_image_on_canvas(input_images[i].resize((718,718)) ,  1024, 1024, alignment='center'), max_size=1024)
                    
                    # Generate with pipeline
                generator = torch.Generator(device=accelerator.device).manual_seed(0)
                results = pipeline(
                        prompt=instruction,
                        input_images=input_images,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=1024,
                        text_guidance_scale=text_guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=1,
                        generator=generator,
                        output_type="pil",
                        width=input_images[0].width,
                        height=input_images[0].height,
                    )
                    
                    # Save generated image
                generated_image = results.images[0]
                save_path = os.path.join(args.output_dir, f"fixed_test_{test_idx}_step_{global_step}.png")
                generated_image.save(save_path)
                    
                    # Create comparison plot
                total_images = len(input_images) + len(results.images)
                fig, axes = plt.subplots(1, total_images, figsize=(5 * total_images, 5))
                    
                if total_images == 1:
                    axes = [axes]  # Make it iterable for single image case
                    
                    # Plot input images
                for i, input_image in enumerate(input_images):
                    axes[i].imshow(input_image)
                    axes[i].axis('off')
                    axes[i].set_title(f'Input {i+1}')
                    
                    # Plot output images
                for i, output_image in enumerate(results.images):
                    axes[len(input_images) + i].imshow(output_image)
                    axes[len(input_images) + i].axis('off')
                    axes[len(input_images) + i].set_title(f'Output {i+1}')
                    
                plt.tight_layout()
                plot_save_path = os.path.join(args.output_dir, f"comparison_test_{test_idx}_step_{global_step}.png")
                plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
                plt.close()
                    
                    # Save instruction
                with open(os.path.join(args.output_dir, f"instruction_test_{test_idx}_step_{global_step}.txt"), "w", encoding='utf-8') as f:
                    f.write(f"Step: {global_step}\nTest Case: {test_idx}\nInstruction: {instruction}\nInput Images: {input_image_paths}")
                    
                logger.info(f"Generated fixed test case {test_idx} at step {global_step}")
                    
                # except Exception as e:
                #     logger.warning(f"Failed to generate test case {test_idx} at step {global_step}: {e}")
                #     continue
        
        pipeline.transformer.train()
    setup_logging(args, accelerator)
    
    # Reproducibility
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # Set performance flags
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)

    model = OmniGen2Transformer2DModel.from_pretrained(
        args.model.pretrained_model_path, subfolder="transformer"
    )
    model.train()

    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        model.config.axes_lens,
        theta=10000,
    )

    if ema_decay != 0:
        model_ema = deepcopy(model)
        model_ema._requires_grad = False

    text_tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_text_encoder_model_name_or_path)
    text_tokenizer.padding_side = "right"

    if accelerator.is_main_process:
        text_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    text_encoder = TextEncoder.from_pretrained(
        args.model.pretrained_text_encoder_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    if args.model.get('resize_token_embeddings', False):
        text_encoder.resize_token_embeddings(len(text_tokenizer))

    if accelerator.is_main_process:
        text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))

    log_model_info("text_encoder", text_encoder)

    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_vae_model_name_or_path,
        subfolder=args.model.get("vae_subfolder", "vae"),
    )
    
    logger.info(vae)
    logger.info("***** Move vae, text_encoder to device and cast to weight_dtype *****")
    # Move vae, unet, text_encoder and controlnet_ema to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    args.train.lora_ft = args.train.get('lora_ft', False)
    if args.train.lora_ft:
        model.requires_grad_(False)

        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # now we will add new LoRA weights the transformer layers
        lora_config = LoraConfig(
            r=args.train.lora_rank,
            lora_alpha=args.train.lora_rank,
            lora_dropout=args.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)

    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    log_model_info("transformer", model)

    # Optimizer creation
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    optimizer = optimizer_class(
        trainable_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")

    with accelerator.main_process_first():
        train_dataset = OmniGen2TrainDataset(
            tokenizer=text_tokenizer,
            use_chat_template=args.data.use_chat_template,
            prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
            ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
            max_input_pixels=OmegaConf.to_object(args.data.get('max_input_pixels', 1024 * 1024)),
            max_output_pixels=args.data.get('max_output_pixels', 1024 * 1024),
            max_side_length=args.data.get('max_side_length', 2048),
            # condition_size=args.data.get('condition_size', (512, 512)),
            # target_size=args.data.get('target_size', (512, 512)),
            # num_shards=args.data.get('num_shards', 1),
        )

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=args.transport.snr_type,
        do_shift=args.transport.do_shift,
        seq_len=args.data.max_output_pixels // 16 // 16,
        dynamic_time_shift=args.transport.get("dynamic_time_shift", False),
        time_shift_version=args.transport.get("time_shift_version", "v1"),
    )  # default: velocity;

    # Log time distribution for analysis
    if accelerator.is_main_process:
        log_time_distribution(transport, accelerator.device, args)

    logger.info(f"Number of training samples: {len(train_dataset)}")

    if args.seed is not None and args.get("workder_specific_seed", False):
        from omnigen2.utils.reproducibility import worker_init_fn

        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    # logger.info("***** Prepare dataLoader *****")
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     batch_size=args.train.batch_size,
    #     num_workers=args.train.dataloader_num_workers,
    #     worker_init_fn=worker_init_fn,
    #     drop_last=True,
    #     collate_fn=OmniGen2Collator(tokenizer=text_tokenizer, max_token_len=args.data.maximum_text_tokens)
    # )

    # logger.info(f"{args.train.batch_size=} {args.train.gradient_accumulation_steps=} {accelerator.num_processes=} {args.train.global_batch_size=}")


    print(f"Accelerate detected {accelerator.num_processes} processes")
    print(f"Torch CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {accelerator.device}")


    assert (
        args.train.batch_size
        * args.train.gradient_accumulation_steps
        * accelerator.num_processes
        == args.train.global_batch_size
    ), (
        f"{args.train.batch_size=} * {args.train.gradient_accumulation_steps=} * {accelerator.num_processes=} should be equal to {args.train.global_batch_size=}"
    )

    # Scheduler and math around the number of training steps.
    

    if args.train.lr_scheduler == 'timm_cosine':
        from omnigen2.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    elif args.train.lr_scheduler == 'timm_constant_with_warmup':
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler

        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t=1,
            decay_rate=1,
            warmup_t=args.train.warmup_t,
            warmup_lr_init=args.train.warmup_lr_init,
            warmup_prefix=args.train.warmup_prefix,
            t_in_epochs=args.train.t_in_epochs,
        )
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps,
            num_training_steps=args.train.max_train_steps,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")


    stages_config = args.train.get('stages', [])
    if not stages_config:
        # Default single stage if no stages defined
        stages_config = [{
            'resolution': 1024,
            'max_pixels': args.data.get('max_output_pixels', 1048576),
            'steps': args.train.max_train_steps
        }]
    
    # Calculate cumulative steps for each stage
    cumulative_steps = []
    total = 0
    for stage in stages_config:
        total += stage['steps']
        cumulative_steps.append(total)
    
    current_stage_idx = 0
    current_stage = stages_config[0]
    
    # Create initial dataloader for first stage
    logger.info(f"Starting Stage 1/{len(stages_config)}: resolution={current_stage['resolution']}, max_pixels={current_stage['max_pixels']}")
    train_dataset, train_dataloader = create_dataloader_for_stage(current_stage, args, text_tokenizer, worker_init_fn)


    if args.train.ema_decay != 0:
        model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader, lr_scheduler
        )
        model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("OmniGen2", init_kwargs={"wandb": {"name": args.name}})

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")




    
    while True:
        
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            print("breaking HEREEE 0")
            break
        
        
        for step, batch in enumerate(train_dataloader):
            # Number of bins, for loss recording
            if (current_stage_idx < len(stages_config) - 1 and 
                global_step >= cumulative_steps[current_stage_idx]):
                
                current_stage_idx += 1
                current_stage = stages_config[current_stage_idx]
                
                logger.info(f"Switching to Stage {current_stage_idx + 1}/{len(stages_config)}: "
                           f"resolution={current_stage['resolution']}, max_pixels={current_stage['max_pixels']}")
                
                # Create new dataloader for new stage
                train_dataset, new_dataloader = create_dataloader_for_stage(current_stage, args, text_tokenizer, worker_init_fn)
                
                # Prepare new dataloader with accelerator
                if args.train.ema_decay != 0:
                    _, _, _, train_dataloader, _ = accelerator.prepare(
                        model, model_ema, optimizer, new_dataloader, lr_scheduler
                    )
                else:
                    _, _, train_dataloader, _ = accelerator.prepare(
                        model, optimizer, new_dataloader, lr_scheduler
                    )
                print("Breaking")
                # Break out of current dataloader loop to start with new one
                break

            n_loss_bins = 10
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device=accelerator.device)
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device=accelerator.device)
            bin_sum_loss = torch.zeros(n_loss_bins, device=accelerator.device)

            input_images = batch['input_images']
            output_image = batch['output_image']
            text_mask = batch['text_mask']
            text_input_ids = batch['text_ids']

            with accelerator.accumulate(model):
                with torch.no_grad():
                    text_feats = text_encoder(
                        input_ids=text_input_ids,
                        attention_mask=text_mask,
                        output_hidden_states=False,
                    ).last_hidden_state

                @torch.no_grad()
                def encode_vae(img):
                    z0 = vae.encode(img.to(dtype=vae.dtype)).latent_dist.sample()
                    if vae.config.shift_factor is not None:
                        z0 = z0 - vae.config.shift_factor
                    if vae.config.scaling_factor is not None:
                        z0 = z0 * vae.config.scaling_factor
                    z0 = z0.to(dtype=weight_dtype)
                    return z0
                
                input_latents = []
                for i, img in enumerate(input_images):
                    if img is not None and len(img) > 0:
                        input_latents.append([])
                        for j, img_j in enumerate(img):
                            input_latents[i].append(encode_vae(img_j).squeeze(0))
                    else:
                        input_latents.append(None)
                
                output_latents = []
                for i, img in enumerate(output_image):
                    output_latents.append(encode_vae(img).squeeze(0))

                model_kwargs = dict(
                    text_hidden_states=text_feats,
                    text_attention_mask=text_mask,
                    ref_image_hidden_states=input_latents,
                    freqs_cis=freqs_cis,
                )

                local_num_tokens_in_batch = 0

                for i, latent in enumerate(output_latents):
                    local_num_tokens_in_batch += latent.numel()
                
                num_tokens_in_batch = accelerator.gather(torch.tensor(local_num_tokens_in_batch, device=accelerator.device)).sum().item()
                loss_dict = transport.training_losses(
                    model,
                    output_latents,
                    model_kwargs,
                    process_index=AcceleratorState().process_index,
                    num_processes=AcceleratorState().num_processes,
                    reduction='sum'
                )
                loss = loss_dict["loss"].sum()
                loss = (loss * accelerator.gradient_state.num_steps * accelerator.num_processes) / num_tokens_in_batch
                total_loss = loss

                accelerator.backward(total_loss)

                bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                detached_loss = loss_dict["loss"].detach()
                
                local_num_tokens = []
                for i, latent in enumerate(output_latents):
                    local_num_tokens.append(latent.numel())
                local_num_tokens = torch.tensor(local_num_tokens, device=accelerator.device)

                # Iterate through each bin index to update occurrence and sum
                for i in range(n_loss_bins):
                    mask = bin_indices == i  # Mask for elements in the i-th bin
                    bin_occurrence[i] = bin_occurrence[i] + local_num_tokens[mask].sum()  # Count occurrences in the i-th bin
                    bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()  # Sum loss values in the i-th bin

                avg_loss = accelerator.gather(loss.detach().repeat(args.train.batch_size)).mean() / accelerator.gradient_state.num_steps
                avg_total_loss = accelerator.gather(total_loss.detach().repeat(args.train.batch_size)).mean() / accelerator.gradient_state.num_steps

                bin_occurrence = accelerator.gather(rearrange(bin_occurrence, "b -> 1 b")).sum(dim=0)
                bin_sum_loss = accelerator.gather(rearrange(bin_sum_loss, "b -> 1 b")).sum(dim=0)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.train.max_grad_norm)

                optimizer.step()
                if 'timm' in args.train.lr_scheduler:
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                logs = {"loss": avg_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                logs["total_loss"] = avg_total_loss.detach().item()

                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        logs[f"loss-bin{i+1}-{n_loss_bins}"] = bin_avg_loss

                if ema_decay != 0:
                    model_ema.step(model.parameters())
                    
                global_step += 1

                if global_step % args.logger.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.logger.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.logger.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                                        
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
                if accelerator.is_main_process:
                    if 'train_visualization_steps' in args.val and (global_step - 1) % args.val.train_visualization_steps == 0:
                        num_samples = min(args.val.get('num_train_visualization_samples', 3), args.train.batch_size)
                        with torch.no_grad():
                            for i in range(num_samples):
                                model_pred = loss_dict['pred'][i]
                                pred_image = loss_dict['xt'][i] + (1 - loss_dict['t'][i]) * model_pred

                                if vae.config.scaling_factor is not None:
                                    pred_image = pred_image / vae.config.scaling_factor
                                if vae.config.shift_factor is not None:
                                    pred_image = pred_image + vae.config.shift_factor
                                pred_image = vae.decode(
                                    pred_image.unsqueeze(0).to(dtype=weight_dtype),
                                    return_dict=False,
                                )[0]
                                pred_image = pred_image.clamp(-1, 1)

                                vis_images = [output_image[i]] + [pred_image]
                                if input_images[i] is not None:
                                    vis_images = input_images[i] + vis_images

                                # Concatenate input images of different sizes horizontally
                                max_height = max(img.shape[-2] for img in vis_images)
                                total_width = sum(img.shape[-1] for img in vis_images)
                                canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)
                                
                                current_x = 0
                                for img in vis_images:
                                    h, w = img.shape[-2:]
                                    # Place image at the top of canvas
                                    canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
                                    current_x += w
                                
                                to_pil_image(canvas).save(os.path.join(args.output_dir, f"input_visualization_{global_step}_{i}_t{loss_dict['t'][i]}.png"))
                                
                                input_ids = text_input_ids[i]
                                instruction = text_tokenizer.decode(input_ids, skip_special_tokens=False)

                                with open(os.path.join(args.output_dir, f"instruction_{global_step}_{i}.txt"), "w", encoding='utf-8') as f:
                                    f.write(f"token len: {len(input_ids)}\ntext: {instruction}")
                    
                    if 'full_generation_steps' in args.val and (global_step-1) % args.val.full_generation_steps == 0:
                        logger.info(f"Generating full images at step {global_step}")
                            
                            
                        scheduler = FlowMatchEulerDiscreteScheduler()
        
                        pipeline = OmniGen2Pipeline(
                            transformer=accelerator.unwrap_model(model),
                            vae=vae,
                            scheduler=scheduler,
                            mllm=text_encoder,  # Assuming this is your MLLM
                            processor=qwen_processor,  # Assuming this works as processor
                        )
                        pipeline = pipeline.to(accelerator.device)
                        
                        generate_full_images_during_training(
                            pipeline=pipeline,
                            global_step=global_step,
                            args=args,
                            num_inference_steps=args.val.get('inference_steps', 50),
                            text_guidance_scale=args.val.get('text_guidance_scale', 5.0),
                            image_guidance_scale=args.val.get('image_guidance_scale', 2.0)
                        )
                        
                        del pipeline

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                print("breaking HEREEE")
                break

    checkpoints = os.listdir(args.output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) > 0 and int(checkpoints[-1].split("-")[1]) < global_step:
        if accelerator.is_main_process:
            if args.logger.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.logger.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)