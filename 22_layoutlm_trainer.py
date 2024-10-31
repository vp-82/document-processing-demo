import os
# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import re
from typing import List, Tuple, Dict
import time
from torch.utils.data import DataLoader

def extract_vendor_name(filename: str) -> str:
    """Extract the actual vendor name without the ID prefix."""
    if filename.startswith(('train_', 'val_')):
        filename = filename[filename.find('_') + 1:]
    
    parts = filename.split('_')
    
    if parts[0].isdigit() or parts[0].startswith(('1', '5')):
        vendor_parts = []
        for part in parts[1:]:
            if part.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')):
                break
            vendor_parts.append(part)
        return '_'.join(vendor_parts)
    
    return parts[0]

class InvoiceTrainer:
    def __init__(self, model_name="microsoft/layoutlmv3-base", dataset_dir="invoice_dataset_processed"):
        self.dataset_dir = Path(dataset_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate dataset directory
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory {dataset_dir} does not exist!")
        
        # Load vendor mapping
        self.load_vendor_mapping()
        
        # Setup device
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and processor
        self.init_model_and_processor(model_name)

    def load_vendor_mapping(self):
        """Load and validate vendor mapping."""
        try:
            with open(self.dataset_dir / 'vendor_map.json', 'r', encoding='utf-8') as f:
                vendor_info = json.load(f)
                self.vendor_map = vendor_info['vendor_map']
                self.vendor_names = vendor_info['vendors']
                
            self.logger.info(f"Loaded {len(self.vendor_names)} vendors:")
            for vendor in self.vendor_names:
                self.logger.info(f"  - {vendor}")
        except Exception as e:
            raise ValueError(f"Error loading vendor mapping: {str(e)}")

    def init_model_and_processor(self, model_name: str):
        """Initialize model and processor with error handling."""
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=True
            )
            
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.vendor_names)
            ).to(self.device)
            
            self.logger.info(f"Model initialized with {len(self.vendor_names)} classes")
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {str(e)}")

    def validate_batch_compatibility(self, dataset, batch_size=2):
        """Validate dataset compatibility before training."""
        self.logger.info("Validating dataset compatibility...")
        
        try:
            # Create a small dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Get first batch
            self.logger.info("Testing batch creation...")
            batch = next(iter(dataloader))
            
            # Log shapes
            self.logger.info("Batch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"{key}: {value.shape}")
            
            # Test forward pass
            self.logger.info("Testing forward pass...")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()})
            
            self.logger.info("Dataset validation successful!")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            self.logger.error("Detailed batch information:")
            if 'batch' in locals():
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        self.logger.error(f"{key}: shape={value.shape}, dtype={value.dtype}")
            raise e

    def process_image(self, image_path: str) -> Dict:
        """Process a single image for the model."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                raise ValueError(f"Unsupported file type: {image_path.suffix}")
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            encoding = self.processor(
                image,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Ensure all tensors are properly shaped
            processed = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Validate tensor shapes
            for key, value in processed.items():
                if isinstance(value, torch.Tensor):
                    if key in ['input_ids', 'attention_mask', 'bbox']:
                        if value.dim() == 1:
                            value = value.unsqueeze(0)
                        if value.shape[0] != 512:
                            self.logger.warning(f"Adjusting {key} shape from {value.shape} to (512, ...)")
                            if value.shape[0] < 512:
                                pad_size = 512 - value.shape[0]
                                padding = torch.zeros(pad_size, *value.shape[1:], dtype=value.dtype)
                                value = torch.cat([value, padding], dim=0)
                            else:
                                value = value[:512]
                        processed[key] = value
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            raise e

    def prepare_dataset(self, image_paths: List[str], labels: List[int]) -> Dataset:
        """Prepare dataset with early validation."""
        self.logger.info(f"Preparing dataset with {len(image_paths)} images")
        
        # Process first image as a test
        self.logger.info("Testing single image processing...")
        try:
            test_processed = self.process_image(image_paths[0])
            self.logger.info("Single image test successful")
            self.logger.info("Test image shapes:")
            for key, value in test_processed.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"{key}: {value.shape}")
        except Exception as e:
            self.logger.error(f"Failed to process test image: {str(e)}")
            raise e

        def process_example(example):
            try:
                processed = self.process_image(example['image_path'])
                processed['labels'] = torch.tensor(example['label'], dtype=torch.long)
                return processed
            except Exception as e:
                self.logger.error(f"Error processing example: {str(e)}")
                raise e

        # Create initial dataset
        dataset = Dataset.from_dict({
            'image_path': image_paths,
            'label': labels
        })
        
        # Process all examples
        processed_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=1,
            desc="Processing images"
        )
        
        # Validate batch compatibility
        self.validate_batch_compatibility(processed_dataset)
        
        return processed_dataset

    def train(self, 
              train_data: List[Tuple[str, int]], 
              eval_data: List[Tuple[str, int]] = None,
              output_dir: str = "trained_model",
              num_epochs: int = 3,
              batch_size: int = 2,
              gradient_accumulation_steps: int = 4):
        """Train with early validation."""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_training_samples': len(train_data),
            'num_eval_samples': len(eval_data) if eval_data else 0,
            'device': str(self.device),
            'model_name': self.model.__class__.__name__,
        }
        
        with open(output_dir / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Test with a small subset first
        self.logger.info("Testing with small subset...")
        test_size = min(5, len(train_data))
        try:
            train_paths, train_labels = zip(*train_data[:test_size])
            test_dataset = self.prepare_dataset(train_paths, train_labels)
            self.logger.info("Subset test successful")
        except Exception as e:
            self.logger.error("Failed subset test")
            raise e

        # Prepare full datasets
        self.logger.info("Processing full dataset...")
        train_paths, train_labels = zip(*train_data)
        train_dataset = self.prepare_dataset(train_paths, train_labels)
        
        eval_dataset = None
        if eval_data:
            self.logger.info("Preparing evaluation dataset...")
            eval_paths, eval_labels = zip(*eval_data)
            eval_dataset = self.prepare_dataset(eval_paths, eval_labels)
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_steps=100,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_data else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_data else False,
            save_total_limit=2,
            logging_dir=str(output_dir / "logs"),
            dataloader_num_workers=0,
            report_to="none",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        start_time = time.time()
        self.logger.info("Starting training...")
        
        try:
            trainer.train()
            
            # Save final model
            self.logger.info("Saving final model...")
            trainer.save_model(str(output_dir / "final_model"))
            self.processor.save_pretrained(str(output_dir / "final_model"))
            
            training_time = time.time() - start_time
            self.logger.info(f"Training complete! Total time: {training_time/60:.2f} minutes")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise e

def main():
    try:
        # Load prepared dataset information
        dataset_dir = Path("invoice_dataset_processed")
        
        if not dataset_dir.exists():
            raise ValueError("Processed dataset directory not found. Please run the PDF conversion script first.")
        
        with open(dataset_dir / 'dataset_stats.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
        print("\nDataset Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Initialize trainer
        trainer = InvoiceTrainer(dataset_dir=dataset_dir)
        
        # Load training and validation data
        def load_data_from_directory(directory):
            data = []
            for file in directory.iterdir():
                if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    try:
                        vendor_name = extract_vendor_name(file.stem)
                        if vendor_name in trainer.vendor_map:
                            data.append((str(file), trainer.vendor_map[vendor_name]))
                        else:
                            print(f"Warning: Unknown vendor '{vendor_name}' from file: {file.name}")
                            print(f"Available vendors: {', '.join(trainer.vendor_names)}")
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
                        continue
            return data
        
        print("\nLoading training data...")
        train_data = load_data_from_directory(dataset_dir / "train")
        print(f"Loaded {len(train_data)} training samples")
        
        print("\nLoading validation data...")
        val_data = load_data_from_directory(dataset_dir / "validation")
        print(f"Loaded {len(val_data)} validation samples")
        
        if not train_data:
            raise ValueError("No training data found!")
        
        # Print some example mappings
        print("\nExample data mappings:")
        for i, (file, label) in enumerate(train_data[:3]):
            print(f"File: {Path(file).name}")
            print(f"Vendor: {trainer.vendor_names[label]}")
            if i >= 2:
                break
        
        # Confirm before training
        response = input("\nDo these mappings look correct? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborting training. Please check the data preparation.")
            return
        
        # Train model
        trainer.train(
            train_data=train_data,
            eval_data=val_data,
            num_epochs=3,
            batch_size=2,
            gradient_accumulation_steps=4,
            output_dir="invoice_model"
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error("Failed to complete training", exc_info=True)

if __name__ == "__main__":
    main()