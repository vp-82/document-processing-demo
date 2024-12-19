@ -0,0 +1,370 @@
# LayoutLM Invoice Classification Tutorial

This tutorial demonstrates how to build an invoice classification system using LayoutLMv3, with detailed explanations of each component.

## Understanding LayoutLMv3
LayoutLMv3 is a multimodal model that can:
- Process document images
- Understand text content
- Comprehend text positioning (layout)
- Combine all these features for document understanding

## Table of Contents
1. [Project Setup](#project-setup)
2. [Data Preparation](#data-preparation)
3. [Model Setup](#model-setup)
4. [Training Pipeline](#training-pipeline)
5. [Validation and Monitoring](#validation-and-monitoring)

## Project Setup

### Required Imports Explained
```python
import os
import torch
from transformers import (
    LayoutLMv3Processor,  # Handles preprocessing of images and text
    LayoutLMv3ForSequenceClassification,  # The actual model
    AdamW  # Optimizer with weight decay
)
from PIL import Image  # For image loading and processing
import numpy as np
from torch.utils.data import Dataset, DataLoader  # For data handling
import wandb  # For experiment tracking

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Configuration Explained
```python
CONFIG = {
    "max_length": 512,    # Maximum sequence length for text tokens
    "batch_size": 1,      # Small batch size for our minimal example
    "num_labels": 2,      # Number of vendor classes
}

train_config = {
    'num_epochs': 3,          # Number of complete passes through the data
    'learning_rate': 1e-5,    # Small learning rate for fine-tuning
}
```

## Data Preparation

### Understanding the Data Structure
The invoice files follow a naming convention:
```
[vendor_id]_[vendor_name]_[invoice_number]_[page].png
Example: 15014330_Shiva_Siegen_320000220000492023_1.png
```

### Vendor Mapping Explained
```python
# Map vendor IDs to names
VENDOR_MAP = {
    "15014330": "Shiva_Siegen",
    "15031152": "Topmech"
}

# Create numerical mappings for labels
# PyTorch models need numerical labels
label2id = {vendor: idx for idx, vendor in enumerate(sorted(set(VENDOR_MAP.values())))}
id2label = {idx: vendor for vendor, idx in label2id.items()}
```

### Dataset Class Explained
```python
class MinimalInvoiceDataset(Dataset):
    """Dataset for invoice classification"""
    def __init__(self, samples, processor):
        self.samples = samples        # List of sample dictionaries
        self.processor = processor    # LayoutLMv3 processor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get single sample
        sample = self.samples[idx]
        
        # Load and convert image to RGB
        image = Image.open(sample['image_path']).convert("RGB")
        
        # Process image through LayoutLMv3 processor
        # This:
        # 1. Extracts text using OCR
        # 2. Gets text positions (bounding boxes)
        # 3. Tokenizes text
        # 4. Normalizes image
        encoding = self.processor(
            image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"]
        )
        
        # Remove batch dimension added by processor
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add label to the encoding
        encoding['labels'] = torch.tensor(sample['label'])
        
        return encoding
```

### Data Loading and Splitting Explained
```python
def create_train_val_split():
    """Create a balanced train/val split"""
    all_samples = []
    
    # Collect all samples from both directories
    for directory in [TRAIN_DATA_DIR, EXTRA_DATA_DIR]:
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                # Extract vendor ID from filename
                vendor_id = filename.split('_')[0]
                vendor_name = VENDOR_MAP[vendor_id]
                
                # Create sample dictionary
                all_samples.append({
                    'image_path': os.path.join(directory, filename),
                    'label': label2id[vendor_name],
                    'vendor_name': vendor_name
                })
    
    # Organize samples by vendor to ensure balanced split
    vendor_samples = {}
    for vendor in set(VENDOR_MAP.values()):
        vendor_samples[vendor] = [s for s in all_samples if s['vendor_name'] == vendor]
    
    # Create train/val split
    train_samples = []
    val_samples = []
    
    for vendor, samples in vendor_samples.items():
        n_total = len(samples)
        # Take up to 4 samples for training, leave at least 1 for validation
        n_train = min(n_total - 1, 4)
        
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:])
    
    return train_samples, val_samples
```

## Model Setup

### Model and Processor Initialization Explained
```python
# Initialize the processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
# The processor handles:
# - Image preprocessing
# - OCR
# - Tokenization
# - Layout analysis

# Initialize the model
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",      # Base pre-trained model
    num_labels=CONFIG["num_labels"],  # Number of vendors to classify
    label2id=label2id,               # Map labels to IDs
    id2label=id2label                # Map IDs back to labels
).to(device)  # Move model to appropriate device (CPU/GPU)
```

## Training Pipeline

### Evaluation Function Explained
```python
def evaluate_model(model, dataloader):
    """Evaluate model performance"""
    model.eval()  # Set model to evaluation mode (disable dropout, etc.)
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch in dataloader:
            # Move batch to model's device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch['labels']
            
            # Get model predictions
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Track accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_predictions, all_labels
```

### Training Functions Explained
```python
def train_single_epoch(model, dataloader, optimizer, epoch):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # Prepare batch
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

## Validation and Monitoring

### Weights & Biases Integration Explained
```python
def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs):
    """Training loop with wandb logging"""
    # Initialize wandb with configuration
    wandb.init(
        project="invoice-classification",
        name="layoutlm-v3-training",
        config={
            "architecture": "LayoutLMv3",
            "dataset": {
                "train_size": len(train_dataloader.dataset),
                "val_size": len(val_dataloader.dataset),
                "num_classes": len(id2label),
                "classes": list(id2label.values())
            },
            "hyperparameters": {
                "learning_rate": train_config['learning_rate'],
                "epochs": num_epochs,
                "batch_size": CONFIG['batch_size'],
                "optimizer": "AdamW"
            }
        }
    )
    
    # Initialize tracking
    best_val_accuracy = 0.0
    predictions_table = wandb.Table(columns=["epoch", "split", "actual", "predicted", "correct"])
    
    for epoch in range(num_epochs):
        # Training Phase
        avg_loss = train_single_epoch(model, train_dataloader, optimizer, epoch)
        
        # Evaluation Phase
        train_accuracy, train_preds, train_labels = evaluate_model(model, train_dataloader)
        val_accuracy, val_preds, val_labels = evaluate_model(model, val_dataloader)
        
        # Log metrics to wandb for visualization
        wandb.log({
            "training": {
                "loss": avg_loss,
                "accuracy": train_accuracy
            },
            "validation": {
                "accuracy": val_accuracy
            },
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
    
    wandb.finish()
    return best_val_accuracy
```

## Understanding the Training Process

### What Happens During Training:

1. **Data Processing**:
   - Images are loaded and converted to RGB
   - LayoutLMv3 processor extracts text and positions
   - Data is converted to tensors for the model

2. **Forward Pass**:
   - Model processes image features
   - Model processes text features
   - Model combines both to make predictions

3. **Backward Pass**:
   - Loss is calculated based on predictions
   - Gradients are computed
   - Model weights are updated

4. **Validation**:
   - Model predictions are compared with true labels
   - Accuracy is calculated
   - Progress is tracked in wandb

### Key Metrics to Monitor:

1. **Training Loss**:
   - Should generally decrease
   - Sudden spikes might indicate learning rate issues

2. **Accuracy**:
   - Training accuracy shows model learning
   - Validation accuracy shows generalization
   - Large gap indicates overfitting

## Common Issues and Solutions

1. **Model Not Learning**:
   - Check learning rate
   - Verify data preprocessing
   - Ensure balanced classes

2. **Overfitting**:
   - Add more training data
   - Implement data augmentation
   - Add regularization

3. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Check image sizes

## Next Steps and Improvements

1. **Data Augmentation**:
   - Rotate images
   - Adjust contrast/brightness
   - Add random noise

2. **Model Improvements**:
   - Try different learning rates
   - Implement early stopping
   - Add regularization

3. **Production Readiness**:
   - Add error handling
   - Implement model saving
   - Create inference pipeline

Would you like any particular section explained in more detail?