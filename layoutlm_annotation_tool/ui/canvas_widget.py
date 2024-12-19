# ui/canvas_widget.py
import tkinter as tk
from typing import Callable, Optional, Tuple, Dict
import logging

from models.annotation import BoundingBox, Annotation
from config.config import Config

logger = logging.getLogger(__name__)

class AnnotationCanvas(tk.Canvas):
    def __init__(self, master, on_box_complete: Callable[[BoundingBox], None], 
                 on_selection_change: Callable[[Optional[str]], None],
                 on_edit_annotation: Callable[[str], None],
                 on_delete_annotation: Callable[[str], None]):
        super().__init__(master)
        self.on_box_complete = on_box_complete
        self.on_selection_change = on_selection_change
        self.current_box = None
        self.drawing = False
        self.start_pos: Optional[Tuple[float, float]] = None
        self.selected_field: Optional[str] = None
        self.on_edit_annotation = on_edit_annotation
        self.on_delete_annotation = on_delete_annotation
        
        # Store references to canvas items
        self.annotation_boxes: Dict[str, int] = {}  # field -> box_id
        self.annotation_labels: Dict[str, int] = {}  # field -> label_id
        
        # Track current field
        self.current_field: str = Config.DEFAULT_FIELDS[0]
        
        # Bind mouse events
        self.bind("<ButtonPress-1>", self.start_box)
        self.bind("<B1-Motion>", self.draw_box)
        self.bind("<ButtonRelease-1>", self.end_box)
        self.bind("<Button-1>", self.handle_click)
        self.bind("<Button-3>", self.show_context_menu)

        self.background_image_id = None
        
        logger.debug("Initialized AnnotationCanvas with selection support")

    def update_display(self):
        """Clear and redraw the canvas."""
        self.delete("all")
        if self.photo:
            # Create background image and store its id
            self.background_image_id = self.create_image(0, 0, anchor="nw", image=self.photo, tags="background")

    def create_image(self, *args, **kwargs):
        """Override create_image to ensure background stays at bottom."""
        item_id = super().create_image(*args, **kwargs)
        self.tag_lower(item_id)  # Push image to back
        return item_id
    
    def show_context_menu(self, event):
        """Show context menu for annotation editing."""
        if not self.selected_field:
            return
            
        menu = Menu(self, tearoff=0)
        menu.add_command(label="Edit", command=lambda: self.on_edit_annotation(self.selected_field))
        menu.add_command(label="Delete", command=lambda: self.on_delete_annotation(self.selected_field))
        menu.post(event.x_root, event.y_root)

    def handle_click(self, event):
        """Handle clicks for selection."""
        if self.drawing:
            return
            
        # Get clicked item
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        items = self.find_overlapping(x-2, y-2, x+2, y+2)
        
        logger.debug(f"Clicked at ({x}, {y}), found items: {items}")
        
        # Filter out background image
        items = [item for item in items if item != self.background_image_id]
        
        if not items:
            # Clicked empty space - deselect
            self.set_selected_field(None)
            return
            
        # Check if clicked item is part of an annotation
        for item_id in items:
            tags = self.gettags(item_id)
            logger.debug(f"Checking tags for item {item_id}: {tags}")
            
            # Look specifically for field_* tags
            for tag in tags:
                if tag.startswith("field_"):
                    # Get the complete field name after "field_"
                    field = tag[6:]  # Remove "field_" prefix
                    logger.debug(f"Found complete field: {field}")
                    self.set_selected_field(field)
                    return
                        
        # If we get here, didn't click an annotation
        self.set_selected_field(None)
    
    def set_selected_field(self, field: Optional[str]) -> None:
        """Set the selected annotation and update visuals."""
        if field == self.selected_field:
            return
            
        logger.debug(f"Canvas changing selection from {self.selected_field} to {field}")
        
        # Update old selection
        if self.selected_field and self.selected_field in self.annotation_boxes:
            self._update_box_style(self.selected_field, False)
        
        # Update new selection
        self.selected_field = field
        if field and field in self.annotation_boxes:
            self._update_box_style(field, True)
            
        # Notify selection change
        self.on_selection_change(field)

    def _update_box_style(self, field: str, is_selected: bool) -> None:
        """Update the visual style of a box based on selection state."""
        if field not in self.annotation_boxes:
            return
            
        box_id = self.annotation_boxes[field]
        color = Config.FIELD_COLORS.get(field, '#000000')
        
        # Determine style based on selection and active states
        is_active = (field == self.current_field)
        width = Config.ACTIVE_BOX_WIDTH if (is_active or is_selected) else Config.INACTIVE_BOX_WIDTH
        dash = None if (is_active or is_selected) else (2, 4)
        
        # Update box style
        self.itemconfig(
            box_id,
            width=width,
            dash=dash,
            outline=color
        )
        
        # If selected, bring to front
        if is_selected:
            self.tag_raise(f"field_{field}")
            self.tag_raise(f"field_{field}_label")
            self.tag_raise(f"field_{field}_label_bg")
    
    def clear_annotations(self):
        """Clear all annotations from canvas."""
        logger.debug("Clearing annotations from canvas")
        # Delete all annotation-related items
        self.delete("annotation")
        self.annotation_boxes.clear()
        self.annotation_labels.clear()
    
    def draw_annotation(self, field: str, annotation: Annotation, is_active: bool = False) -> None:
        """Draw a single annotation with its label."""
        logger.debug(f"Drawing annotation - Field: {field}, Active: {is_active}")
        
        # Get color for this field
        color = Config.FIELD_COLORS.get(field, '#000000')
        
        # Set properties based on both active and selection states
        is_selected = (field == self.selected_field)
        width = Config.ACTIVE_BOX_WIDTH if (is_active or is_selected) else Config.INACTIVE_BOX_WIDTH
        dash = None if (is_active or is_selected) else (2, 4)
        
        try:
            # Create rectangle for annotation
            box_id = self.create_rectangle(
                *annotation.box.coordinates,
                outline=color,
                width=width,
                dash=dash,
                tags=("annotation", f"field_{field}"),
                stipple='gray50' if not (is_active or is_selected) else ''
            )
            self.annotation_boxes[field] = box_id
            
            # Create text label with common field tag
            label_bg = self.create_rectangle(
                annotation.box.x1, annotation.box.y1 - 20,
                annotation.box.x1 + len(field) * 8 + 10, annotation.box.y1,
                fill=Config.LABEL_BG,
                outline=color,
                tags=("annotation", f"field_{field}")  # Same tag as box
            )
            
            label_id = self.create_text(
                annotation.box.x1 + 5, annotation.box.y1 - 10,
                text=field,
                anchor='w',
                font=Config.LABEL_FONT,
                fill=color,
                tags=("annotation", f"field_{field}")  # Same tag as box
            )
            self.annotation_labels[field] = label_id
            
            # Ensure annotations are above background
            self.tag_raise("annotation")
            
            logger.debug(f"Successfully drew annotation for field {field}")
        except Exception as e:
            logger.error(f"Error drawing annotation: {str(e)}")
    
    def update_annotations(self, annotations: Dict[str, Annotation]) -> None:
        """Update all annotations on canvas."""
        logger.debug(f"Updating annotations on canvas. Count: {len(annotations)}")
        self.clear_annotations()
        
        # Draw all annotations
        for field, annotation in annotations.items():
            is_active = (field == self.current_field)
            self.draw_annotation(field, annotation, is_active)
    
    def set_current_field(self, field: str) -> None:
        """Update current field and refresh annotations display."""
        logger.debug(f"Setting current field to: {field}")
        old_field = self.current_field
        self.current_field = field
        
        # Update appearance of affected annotations
        if old_field in self.annotation_boxes:
            self.itemconfig(
                self.annotation_boxes[old_field],
                width=Config.INACTIVE_BOX_WIDTH,
                dash=(2, 4),
                stipple='gray50'
            )
        
        if field in self.annotation_boxes:
            self.itemconfig(
                self.annotation_boxes[field],
                width=Config.ACTIVE_BOX_WIDTH,
                dash=None,
                stipple=''
            )
    
    def start_box(self, event):
        logger.debug(f"Starting box at ({event.x}, {event.y})")
        self.drawing = True
        self.start_pos = (self.canvasx(event.x), self.canvasy(event.y))
        
        if self.current_box:
            self.delete(self.current_box)
    
    def draw_box(self, event):
        if not self.drawing or not self.start_pos:
            return
            
        if self.current_box:
            self.delete(self.current_box)
        
        cur_x = self.canvasx(event.x)
        cur_y = self.canvasy(event.y)
        
        color = Config.FIELD_COLORS.get(self.current_field, '#000000')
        self.current_box = self.create_rectangle(
            self.start_pos[0], self.start_pos[1],
            cur_x, cur_y,
            outline=color,
            width=Config.ACTIVE_BOX_WIDTH
        )
    
    def end_box(self, event):
        if not self.drawing or not self.start_pos:
            logger.debug("Box drawing ended but no active drawing")
            return
            
        self.drawing = False
        end_x = self.canvasx(event.x)
        end_y = self.canvasy(event.y)
        
        # Check if box is too small
        if abs(end_x - self.start_pos[0]) < 5 or abs(end_y - self.start_pos[1]) < 5:
            logger.debug("Box too small, ignoring")
            return
            
        box = BoundingBox(
            x1=min(self.start_pos[0], end_x),
            y1=min(self.start_pos[1], end_y),
            x2=max(self.start_pos[0], end_x),
            y2=max(self.start_pos[1], end_y)
        )
        
        logger.debug(f"Box complete: ({box.x1}, {box.y1}) to ({box.x2}, {box.y2})")
        logger.debug("Calling on_box_complete callback")
        self.on_box_complete(box)