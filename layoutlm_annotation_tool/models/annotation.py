# models/annotation.py
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Tuple
import logging

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_dict(cls, data: List[float]) -> 'BoundingBox':
        return cls(x1=data[0], y1=data[1], x2=data[2], y2=data[3])

@dataclass
class Annotation:
    field: str
    box: BoundingBox
    text: str
    page: int

    def to_dict(self) -> dict:
        return {
            'box': self.box.to_dict(),
            'text': self.text,
            'field': self.field
        }
    
    @classmethod
    def from_dict(cls, data: dict, field_name: str, page: int) -> 'Annotation':
        return cls(
            field=field_name,
            box=BoundingBox.from_dict(data['box']),
            text=data['text'],
            page=page
        )

@dataclass
class AnnotationStore:
    annotations: Dict[int, Dict[str, Annotation]] = dataclass_field(default_factory=dict)
    
    def add_annotation(self, field: str, box: BoundingBox, text: str, page: int) -> None:
        """Add or update an annotation."""
        logging.debug(f"Adding annotation - Field: {field}, Page: {page}, Text: {text}")
        if page not in self.annotations:
            self.annotations[page] = {}
        self.annotations[page][field] = Annotation(field, box, text, page)
        logging.debug(f"Current annotations: {self.annotations}")
    
    def get_page_annotations(self, page: int) -> Dict[str, Annotation]:
        """Get all annotations for a specific page."""
        return self.annotations.get(page, {})

    def clear(self) -> None:
        """Clear all annotations."""
        logging.debug("Clearing all annotations")
        self.annotations.clear()

    def to_dict(self) -> dict:
        """Convert annotations to dictionary format for saving."""
        result = {}
        for page, page_anns in self.annotations.items():
            # Create page entry if it doesn't exist
            if str(page) not in result:
                result[str(page)] = {}
            
            # Add annotations for each field
            for field_name, annotation in page_anns.items():
                result[str(page)][field_name] = annotation.to_dict()
        
        logging.debug(f"Serializing annotations: {result}")
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnnotationStore':
        """Create AnnotationStore from dictionary data."""
        store = cls()
        for page_str, page_data in data.items():
            page = int(page_str)
            for field_name, ann_data in page_data.items():
                box = BoundingBox.from_dict(ann_data['box'])
                store.add_annotation(
                    field=field_name,
                    box=box,
                    text=ann_data['text'],
                    page=page
                )
        return store
    
@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_dict(cls, data: List[float]) -> 'BoundingBox':
        return cls(x1=data[0], y1=data[1], x2=data[2], y2=data[3])

    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        """Return coordinates as a tuple for canvas drawing."""
        return (self.x1, self.y1, self.x2, self.y2)