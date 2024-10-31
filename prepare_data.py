import os
from pathlib import Path
import pdf2image
import logging
from tqdm import tqdm
import shutil
import json

def extract_vendor_name(filename: str) -> str:
    """Extract the actual vendor name without the ID prefix."""
    parts = filename.split('_')
    
    # The first part is typically the ID number
    if parts[0].isdigit() or parts[0].startswith(('1', '5')):
        # Extract vendor name (skip ID and document number/date)
        vendor_parts = []
        for part in parts[1:]:
            # Stop when we hit what looks like a document number
            if part.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')):
                break
            vendor_parts.append(part)
        return '_'.join(vendor_parts)
    
    return parts[0]

class DocumentPreprocessor:
    def __init__(self, input_dir="./Data", output_dir="invoice_dataset_processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "validation").mkdir(exist_ok=True)
        
        # Create vendor mapping
        self.create_vendor_mapping()

    def create_vendor_mapping(self):
        """Create mapping of vendor names without ID prefixes."""
        self.vendor_names = set()
        
        # Scan input directories for vendor names
        for dir_path in self.input_dir.iterdir():
            if dir_path.is_dir():
                vendor_name = extract_vendor_name(dir_path.name)
                self.vendor_names.add(vendor_name)
        
        # Create sorted list and mapping
        self.vendor_names = sorted(self.vendor_names)
        self.vendor_map = {name: idx for idx, name in enumerate(self.vendor_names)}
        
        # Save vendor mapping
        with open(self.output_dir / 'vendor_map.json', 'w') as f:
            json.dump({
                'vendor_map': self.vendor_map,
                'vendors': self.vendor_names
            }, f, indent=2)
            
        self.logger.info(f"Created vendor mapping for {len(self.vendor_names)} vendors:")
        for vendor in self.vendor_names:
            self.logger.info(f"  - {vendor}")

    def convert_pdf_to_image(self, pdf_path):
        """Convert a single PDF to image."""
        try:
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=200,
                fmt="png"
            )
            return images[0]
        except Exception as e:
            self.logger.error(f"Error converting {pdf_path}: {str(e)}")
            raise e

    def process_directory(self, src_dir, dest_dir, split_ratio=0.2):
        """Process all files in a directory."""
        src_dir = Path(src_dir)
        
        # Get list of files
        files = list(src_dir.glob("*.*"))
        
        for file_path in tqdm(files, desc=f"Processing {src_dir.name}"):
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Convert PDF to image
                    image = self.convert_pdf_to_image(file_path)
                    # Save as PNG with vendor name in filename
                    vendor_name = extract_vendor_name(src_dir.name)
                    output_name = f"{src_dir.name}_{file_path.stem}.png"
                    
                    # Decide whether to put in train or validation
                    if hash(str(file_path)) % 100 < split_ratio * 100:
                        output_path = self.output_dir / "validation" / output_name
                    else:
                        output_path = self.output_dir / "train" / output_name
                        
                    image.save(output_path, "PNG")
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    def process_dataset(self, split_ratio=0.2):
        """Process entire dataset."""
        self.logger.info("Starting dataset conversion...")
        
        # Process each vendor directory
        for vendor_dir in self.input_dir.iterdir():
            if vendor_dir.is_dir():
                self.logger.info(f"Processing vendor: {vendor_dir.name}")
                self.process_directory(vendor_dir, self.output_dir, split_ratio)
        
        # Generate dataset statistics
        train_files = list((self.output_dir / "train").glob("*.png"))
        val_files = list((self.output_dir / "validation").glob("*.png"))
        
        stats = {
            'num_classes': len(self.vendor_names),
            'vendors': self.vendor_names,
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'vendor_mapping': self.vendor_map
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Dataset conversion complete!")
        self.logger.info(f"Created {len(train_files)} training samples")
        self.logger.info(f"Created {len(val_files)} validation samples")

def main():
    # First, make sure poppler is installed
    try:
        import pdf2image
    except ImportError:
        print("Please install pdf2image and poppler first:")
        print("  pip install pdf2image")
        print("  brew install poppler  # For Mac")
        return
        
    # Initialize preprocessor
    processor = DocumentPreprocessor(
        input_dir="./Data",
        output_dir="invoice_dataset_processed"
    )
    
    # Process the dataset
    processor.process_dataset(split_ratio=0.2)
    
    print("\nConversion complete! You can now use 'invoice_dataset_processed' for training.")

if __name__ == "__main__":
    main()