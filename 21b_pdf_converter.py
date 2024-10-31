import os
from pathlib import Path
import pdf2image
import logging
from tqdm import tqdm
import shutil

class DocumentPreprocessor:
    def __init__(self, input_dir="invoice_dataset", output_dir="invoice_dataset_processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "validation").mkdir(exist_ok=True)
        
    def convert_pdf_to_image(self, pdf_path):
        """Convert a single PDF to image."""
        try:
            # Convert PDF to image
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=200,  # Adjust DPI as needed
                fmt="png"
            )
            
            # We'll use only the first page for training
            return images[0]
            
        except Exception as e:
            self.logger.error(f"Error converting {pdf_path}: {str(e)}")
            raise e

    def process_directory(self, src_dir, dest_dir):
        """Process all files in a directory."""
        src_dir = Path(src_dir)
        dest_dir = Path(dest_dir)
        
        # Get list of files
        files = list(src_dir.glob("*.*"))
        
        for file_path in tqdm(files, desc=f"Processing {src_dir.name}"):
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Convert PDF to image
                    image = self.convert_pdf_to_image(file_path)
                    # Save as PNG with same base name
                    output_path = dest_dir / f"{file_path.stem}.png"
                    image.save(output_path, "PNG")
                else:
                    # Copy non-PDF files directly
                    shutil.copy2(file_path, dest_dir / file_path.name)
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    def copy_metadata(self):
        """Copy metadata files to new directory."""
        metadata_files = ['vendor_map.json', 'dataset_stats.json']
        for file in metadata_files:
            if (self.input_dir / file).exists():
                shutil.copy2(self.input_dir / file, self.output_dir / file)

    def process_dataset(self):
        """Process entire dataset."""
        self.logger.info("Starting dataset conversion...")
        
        # Process train directory
        self.logger.info("Processing training data...")
        self.process_directory(
            self.input_dir / "train",
            self.output_dir / "train"
        )
        
        # Process validation directory
        self.logger.info("Processing validation data...")
        self.process_directory(
            self.input_dir / "validation",
            self.output_dir / "validation"
        )
        
        # Copy metadata files
        self.copy_metadata()
        
        self.logger.info("Dataset conversion complete!")
        
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
        input_dir="invoice_dataset",
        output_dir="invoice_dataset_processed"
    )
    
    # Process the dataset
    processor.process_dataset()
    
    print("\nConversion complete! You can now use 'invoice_dataset_processed' for training.")

if __name__ == "__main__":
    main()