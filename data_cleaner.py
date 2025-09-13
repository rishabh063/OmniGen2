#!/usr/bin/env python3
"""
Dataset Processing Script
Processes images from Subjects200K dataset, detects objects, and saves processed versions.
"""

import os
import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, output_dir="processed_dataset", cache_dir="./cache"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset = None
        self.model = None
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self):
        """Load and filter the dataset"""
        print("Loading dataset...")
        dataset = load_dataset('Yuanshi/Subjects200K')
        
        def filter_func(item):
            if item.get("collection") != "collection_2":
                return False
            if not item.get("quality_assessment"):
                return False
            return all(
                item["quality_assessment"].get(key, 0) >= 5
                for key in ["compositeStructure", "objectConsistency", "imageQuality"]
            )
        
        self.dataset = dataset["train"].filter(
            filter_func,
            num_proc=16,
            cache_file_name=str(self.cache_dir / "collection_2_valid.arrow")
        )
        print(f"Dataset loaded with {len(self.dataset)} items")
    
    def load_model(self):
        """Load the moondream2 model"""
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": "cuda"}  # Change to 'mps' for Apple Silicon
        )
        print("Model loaded successfully")
    
    def scalebbox(self, imgsize, bboxes):
        """Scale bounding boxes to image dimensions"""
        img_width, img_height = imgsize
        boxarray = []
        for bbox in bboxes:
            x_min = int(bbox['x_min'] * img_width)
            y_min = int(bbox['y_min'] * img_height)
            x_max = int(bbox['x_max'] * img_width)
            y_max = int(bbox['y_max'] * img_height)
            boxarray.append([x_min, y_min, x_max, y_max])
        return boxarray
    
    def draw_bboxes(self, image, bboxes, box_color='red', line_width=2):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        if isinstance(bboxes[0], (int, float)):
            bboxes = [bboxes]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)
        
        return img_copy
    
    def black_out_bboxes(self, image, bboxes):
        """Black out regions defined by bounding boxes"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        if isinstance(bboxes[0], (int, float)):
            bboxes = [bboxes]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], fill='black')
        
        return img_copy
    
    def bigbbox(self, img, bboxes):
        """Find the bounding box that encompasses all given bboxes"""
        min_x, min_y = img.size
        max_x, max_y = 0, 0
        for bbox in bboxes:
            if bbox[0] < min_x:
                min_x = bbox[0]
            if bbox[1] < min_y:
                min_y = bbox[1]
            if bbox[2] > max_x:
                max_x = bbox[2]
            if bbox[3] > max_y:
                max_y = bbox[3]
        return [min_x, min_y, max_x, max_y]
    
    def addPadding(self, img, bboxes, padding):
        """Add padding to bounding boxes"""
        for idx in range(len(bboxes)):
            min_x, min_y, max_x, max_y = bboxes[idx]
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(img.size[0], max_x + padding)
            max_y = min(img.size[1], max_y + padding)
            bboxes[idx] = [min_x, min_y, max_x, max_y]
        return bboxes
    
    def create_images(self, img, bboxes, padding=10):
        """Create reference and mask images"""
        bboxes_copy = [bbox.copy() for bbox in bboxes]
        
        if random.random() < 0.5:
            big_bbox = self.bigbbox(img, bboxes_copy)
            ref_img = img.crop(big_bbox)
        else:
            big_bbox = self.bigbbox(img, self.addPadding(img, bboxes_copy, padding))
            ref_img = img.crop(big_bbox)
        
        if random.random() < 0.5:
            mask_image = self.black_out_bboxes(img, bboxes)
        else:
            tempbboxes = self.addPadding(img, [bbox.copy() for bbox in bboxes], padding)
            mask_image = self.black_out_bboxes(img, tempbboxes)
        
        return ref_img, mask_image
    
    def process_single_item(self, index, padding=8, size=512):
        """Process a single dataset item"""
        item = self.dataset[index]
        category = item['description']['item']
        
        # Create folder for this index
        item_dir = self.output_dir / f"idx_{index:06d}"
        item_dir.mkdir(exist_ok=True)
        
        # Save original image
        original_img = item['image']
        original_img.save(item_dir / f"original_{category.replace(' ', '_')}.png")
        
        # Crop two regions
        img1 = original_img.crop((padding, padding, size + padding, size + padding))
        img2 = original_img.crop((size + 2*padding, padding, 2*size + 2*padding, size + padding))
        
        # Save cropped images
        img1.save(item_dir / f"crop1_{category.replace(' ', '_')}.png")
        img2.save(item_dir / f"crop2_{category.replace(' ', '_')}.png")
        
        # Process first crop
        objects1 = self.scalebbox(img1.size, self.model.detect(img1, category)["objects"])
        if len(objects1) >= 1:
            ref_img1, mask_img1 = self.create_images(img1, objects1, padding)
            ref_img1.save(item_dir / f"ref1_{category.replace(' ', '_')}.png")
            mask_img1.save(item_dir / f"mask1_{category.replace(' ', '_')}.png")
        
        # Process second crop
        objects2 = self.scalebbox(img2.size, self.model.detect(img2, category)["objects"])
        if len(objects2) >= 1:
            ref_img2, mask_img2 = self.create_images(img2, objects2, padding)
            ref_img2.save(item_dir / f"ref2_{category.replace(' ', '_')}.png")
            mask_img2.save(item_dir / f"mask2_{category.replace(' ', '_')}.png")
        
        # Save metadata
        metadata = {
            'index': index,
            'category': category,
            'objects1_count': len(objects1),
            'objects2_count': len(objects2),
            'objects1': objects1,
            'objects2': objects2
        }
        
        with open(item_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(objects1) > 0 or len(objects2) > 0
    
    def get_processed_indices(self):
        """Get list of already processed indices"""
        processed = set()
        if self.output_dir.exists():
            for item_dir in self.output_dir.iterdir():
                if item_dir.is_dir() and item_dir.name.startswith('idx_'):
                    try:
                        idx = int(item_dir.name.split('_')[1])
                        processed.add(idx)
                    except (ValueError, IndexError):
                        continue
        return processed
    
    def process_dataset(self, start_idx=0, end_idx=None, skip_existing=True):
        """Process dataset items from start_idx to end_idx"""
        if self.dataset is None:
            self.load_dataset()
        if self.model is None:
            self.load_model()
        
        if end_idx is None:
            end_idx = len(self.dataset)
        
        processed_indices = self.get_processed_indices() if skip_existing else set()
        
        total_processed = 0
        successful_processed = 0
        
        indices_to_process = [i for i in range(start_idx, min(end_idx, len(self.dataset))) 
                             if i not in processed_indices]
        
        if not indices_to_process:
            print("No new indices to process!")
            return
        
        print(f"Processing {len(indices_to_process)} items (indices {start_idx} to {end_idx-1})")
        if skip_existing and processed_indices:
            print(f"Skipping {len(processed_indices)} already processed items")
        
        for idx in tqdm(indices_to_process, desc="Processing images"):
            try:
                success = self.process_single_item(idx)
                total_processed += 1
                if success:
                    successful_processed += 1
            except Exception as e:
                print(f"\nError processing index {idx}: {str(e)}")
                continue
        
        print(f"\nCompleted! Processed {total_processed} items, {successful_processed} with detected objects")

def main():
    parser = argparse.ArgumentParser(description='Process Subjects200K dataset')
    parser.add_argument('--output_dir', type=str, default='processed_dataset',
                       help='Output directory for processed images')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending index (default: process all)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory for dataset')
    parser.add_argument('--no_skip_existing', action='store_true',
                       help='Process already existing indices (default: skip existing)')
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    processor.process_dataset(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_existing=not args.no_skip_existing
    )

if __name__ == "__main__":
    main()




