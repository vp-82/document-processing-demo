import os
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
import logging
import pandas as pd

class InvoiceDataPreparator:
    def __init__(self, base_dir="./Data", output_dir="invoice_dataset"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure immediately
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create all necessary directories."""
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train and validation directories
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directory structure in {self.output_dir}")

    def scan_invoice_directories(self):
        """Scan directories and collect invoice files."""
        invoice_files = []
        vendor_names = []
        
        # Check if base directory exists
        if not self.base_dir.exists():
            raise ValueError(f"Base directory {self.base_dir} does not exist")
        
        for vendor_dir in self.base_dir.iterdir():
            if vendor_dir.is_dir():
                vendor_name = vendor_dir.name
                vendor_names.append(vendor_name)
                
                # Scan for invoice files
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf']:
                    for file_path in vendor_dir.glob(ext):
                        invoice_files.append({
                            'path': str(file_path),
                            'vendor': vendor_name,
                            'filename': file_path.name
                        })
        
        if not invoice_files:
            self.logger.warning("No invoice files found!")
            self.logger.info(f"Searched in directory: {self.base_dir}")
            self.logger.info("Supported extensions: .png, .jpg, .jpeg, .pdf")
            
        return invoice_files, sorted(vendor_names)

    def prepare_dataset(self, test_size=0.2, random_state=42):
        """Prepare the dataset from invoice files."""
        # Scan for invoice files
        self.logger.info("Scanning for invoice files...")
        invoice_files, vendor_names = self.scan_invoice_directories()
        
        if not invoice_files:
            raise ValueError("No invoice files found in the specified directories")
        
        self.logger.info(f"Found {len(invoice_files)} files from {len(vendor_names)} vendors")
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(invoice_files)
        
        # Create vendor to label mapping
        vendor_map = {vendor: idx for idx, vendor in enumerate(vendor_names)}
        
        # Save vendor mapping
        vendor_map_path = self.output_dir / 'vendor_map.json'
        with open(vendor_map_path, 'w') as f:
            json.dump({
                'vendor_map': vendor_map,
                'vendors': vendor_names
            }, f, indent=2)
        
        self.logger.info(f"Saved vendor mapping to {vendor_map_path}")
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['vendor']
        )
        
        # Prepare data structures for training
        train_data = []
        val_data = []
        
        # Process training files
        self.logger.info("Copying training files...")
        for _, row in train_df.iterrows():
            dest_path = self.train_dir / f"{row['vendor']}_{row['filename']}"
            shutil.copy2(row['path'], dest_path)
            train_data.append((str(dest_path), vendor_map[row['vendor']]))
        
        # Process validation files
        self.logger.info("Copying validation files...")
        for _, row in val_df.iterrows():
            dest_path = self.val_dir / f"{row['vendor']}_{row['filename']}"
            shutil.copy2(row['path'], dest_path)
            val_data.append((str(dest_path), vendor_map[row['vendor']]))
        
        # Save dataset statistics
        stats = {
            'num_classes': len(vendor_names),
            'vendors': vendor_names,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'class_distribution': {
                'train': train_df['vendor'].value_counts().to_dict(),
                'validation': val_df['vendor'].value_counts().to_dict()
            }
        }
        
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset prepared: {len(train_data)} training samples, {len(val_data)} validation samples")
        self.logger.info(f"Number of vendor types: {len(vendor_names)}")
        
        return train_data, val_data, vendor_names

def main():
    # Initialize preparator with your data directory
    preparator = InvoiceDataPreparator(
        base_dir="./Data",
        output_dir="invoice_dataset"
    )
    
    try:
        # Prepare the dataset
        train_data, val_data, vendor_names = preparator.prepare_dataset(
            test_size=0.2  # 20% for validation
        )
        
        # Print dataset summary
        print("\nDataset Summary:")
        with open(Path("invoice_dataset") / 'dataset_stats.json', 'r') as f:
            stats = json.load(f)
            print(json.dumps(stats, indent=2))
        
        # Print example of training data structure
        print("\nExample training data entries:")
        for i, (path, label) in enumerate(train_data[:3]):
            print(f"File: {path}")
            print(f"Vendor Label: {label} ({vendor_names[label]})")
            if i >= 2:
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Failed to prepare dataset: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()