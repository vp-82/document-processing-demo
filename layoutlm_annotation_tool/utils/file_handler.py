# utils/file_handler.py
import json
import os
import logging
from typing import Optional

from models.annotation import AnnotationStore, BoundingBox, Annotation

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def get_annotation_path(image_path: str) -> str:
        """Get the corresponding annotation file path for an image."""
        base_path = os.path.splitext(image_path)[0]
        return f"{base_path}_annotations.json"

    @staticmethod
    def save_annotations(image_path: str, annotations: AnnotationStore) -> None:
        """Save annotations to a JSON file."""
        json_path = FileHandler.get_annotation_path(image_path)
        
        data = {
            'image_path': image_path,
            'annotations': annotations.to_dict()
        }
        
        logger.debug(f"Saving annotations to {json_path}: {data}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Annotations saved to: {json_path}")
    
    @staticmethod
    def load_annotations(image_path: str) -> Optional[AnnotationStore]:
        """Load annotations from a JSON file."""
        json_path = FileHandler.get_annotation_path(image_path)
        
        if not os.path.exists(json_path):
            logger.debug(f"No annotation file found at {json_path}")
            return None
            
        try:
            logger.debug(f"Loading annotations from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            store = AnnotationStore()
            annotations_dict = data.get('annotations', {})
            
            # Convert the loaded data back into annotations
            for page_str, page_data in annotations_dict.items():
                page = int(page_str)
                for field, field_data in page_data.items():
                    box = BoundingBox(*field_data['box'])
                    text = field_data['text']
                    store.add_annotation(
                        field=field,
                        box=box,
                        text=text,
                        page=page
                    )
            
            logger.debug(f"Successfully loaded annotations: {store.annotations}")
            return store
            
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            return None