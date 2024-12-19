# utils/ocr_handler.py
import pytesseract
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class OCRHandler:
    @staticmethod
    def extract_text(image: Image.Image, box: tuple) -> str:
        """
        Extract text from the specified area of the image.
        Optimized for multi-line text blocks.
        
        Args:
            image: PIL Image object
            box: Tuple of (x1, y1, x2, y2) coordinates
        
        Returns:
            str: Extracted text or empty string if extraction fails
        """
        try:
            # Crop the image to the box area
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            
            # Convert to OpenCV format for preprocessing
            cv_image = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            
            # Enhanced preprocessing for text blocks
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # C constant
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Convert back to PIL Image
            preprocessed = Image.fromarray(denoised)
            
            # Extract text with configuration for block text
            text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 6 --oem 3'  # PSM 6 is for uniform block of text
            ).strip()
            
            logger.debug(f"Extracted text from box: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""