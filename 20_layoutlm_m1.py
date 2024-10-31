import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import os

class M1DocumentProcessor:
    def __init__(self, model_name="microsoft/layoutlmv3-base", num_labels=5):
        # Check if MPS is available
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=True
        )
        
        # Load model and convert to float32
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Convert model to float32 before moving to device
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        
    def process_document(self, image_path):
        """Process a document image with consistent tensor types."""
        # Load and resize image if needed
        image = Image.open(image_path).convert("RGB")
        max_size = 1600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        try:
            # Process the image
            encoding = self.processor(
                image,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Handle tensor types consistently
            for k, v in encoding.items():
                if torch.is_tensor(v):
                    if k in ['input_ids', 'attention_mask', 'bbox']:
                        # Integer tensors
                        encoding[k] = v.to(self.device, dtype=torch.long)
                    else:
                        # Float tensors
                        encoding[k] = v.to(self.device, dtype=torch.float32)
            
            return encoding
            
        except RuntimeError as e:
            print(f"Error during processing: {str(e)}")
            raise e

    def classify_document(self, image_path):
        """Classify a document with consistent tensor handling."""
        # Process document
        encoding = self.process_document(image_path)
        
        # Run inference in eval mode
        try:
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            return predictions.cpu().numpy()
        except Exception as e:
            print(f"Classification error: {str(e)}")
            print(f"Tensor types in encoding:")
            for k, v in encoding.items():
                if torch.is_tensor(v):
                    print(f"{k}: {v.dtype}")
            raise e

def setup_m1_environment():
    """Setup the M1 environment with proper configurations."""
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

def main():
    # Setup environment
    setup_m1_environment()
    
    # Initialize processor
    processor = M1DocumentProcessor(num_labels=3)  # Adjust based on your classes
    
    # Process a document
    image_path = "Test_Invoice.png"
    try:
        predictions = processor.classify_document(image_path)
        print(f"Classification probabilities: {predictions}")
    except Exception as e:
        print(f"Error during classification: {str(e)}")

if __name__ == "__main__":
    main()