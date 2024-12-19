# ui/main_window.py
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('annotation_tool.log')
    ]
)
logger = logging.getLogger(__name__)

from config.config import Config
from models.annotation import AnnotationStore, BoundingBox, Annotation
from models.image_data import ImageData
from ui.canvas_widget import AnnotationCanvas 
from ui.control_panel import ControlPanel
from ui.dialogs import TextInputDialog
from utils.file_handler import FileHandler
from utils.image_utils import ImageUtils
from utils.ocr_handler import OCRHandler  # Add this import

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        logger.debug("Initializing MainWindow")
        self.title(Config.DEFAULT_WINDOW_TITLE)
        
        # Initialize data
        self.current_image: Optional[ImageData] = None
        self.annotation_store = AnnotationStore()
        self.photo = None
        self.selected_field: Optional[str] = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = AnnotationCanvas(
            canvas_frame, 
            on_box_complete=self.handle_box_complete,
            on_selection_change=self.handle_selection_change,
            on_edit_annotation=self.handle_edit_annotation,
            on_delete_annotation=self.handle_delete_annotation
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.control_panel = ControlPanel(
            self,
            on_load_image=self.load_image,
            on_save=self.save_annotations,
            on_prev_page=self.prev_page,
            on_next_page=self.next_page,
            on_field_change=self.handle_field_change,
            on_selection_change=self.handle_selection_change,
            on_edit_annotation=self.handle_edit_annotation,
            on_delete_annotation=self.handle_delete_annotation
        )
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)

    def handle_delete_annotation(self, field: str) -> None:
        """Handle deletion of an annotation."""
        if not self.current_image:
            return
        
        try:
            logger.debug(f"Deleting annotation for field: {field}")
            # Get current page annotations
            page = self.current_image.current_page_idx
            current_annotations = self.annotation_store.get_page_annotations(page)
            
            # Remove the annotation
            if field in current_annotations:
                del current_annotations[field]
                
                # Update display
                self.canvas.update_annotations(current_annotations)
                self.control_panel.update_annotation_list(current_annotations)
                
                # Clear selection
                self.handle_selection_change(None)
                
                logger.debug(f"Successfully deleted annotation for field: {field}")
        except Exception as e:
            logger.error(f"Error deleting annotation: {str(e)}")
            messagebox.showerror(
                "Delete Error",
                f"Failed to delete annotation: {str(e)}"
            )

    def handle_edit_annotation(self, field: str) -> None:
        """Handle editing of an annotation."""
        if not self.current_image:
            return
        
        try:
            logger.debug(f"Editing annotation for field: {field}")
            current_annotations = self.annotation_store.get_page_annotations(
                self.current_image.current_page_idx
            )
            
            if field in current_annotations:
                annotation = current_annotations[field]
                # Show dialog with current text
                new_text = TextInputDialog.get_text(
                    self,
                    field,
                    initial_text=annotation.text
                )
                
                if new_text:
                    # Update annotation with new text
                    self.annotation_store.add_annotation(
                        field=field,
                        box=annotation.box,
                        text=new_text,
                        page=self.current_image.current_page_idx
                    )
                    
                    # Update display
                    current_annotations = self.annotation_store.get_page_annotations(
                        self.current_image.current_page_idx
                    )
                    self.canvas.update_annotations(current_annotations)
                    self.control_panel.update_annotation_list(current_annotations)
                    
                    logger.debug(f"Successfully updated annotation for field: {field}")
        except Exception as e:
            logger.error(f"Error editing annotation: {str(e)}")
            messagebox.showerror(
                "Edit Error",
                f"Failed to edit annotation: {str(e)}"
            )

    def handle_selection_change(self, field: Optional[str]) -> None:
        """Handle selection changes from both canvas and control panel."""
        logger.debug(f"Selection changed to: {field}")
        if field == self.selected_field:
            return

        self.selected_field = field
        
        # Sync canvas selection
        if hasattr(self.canvas, 'set_selected_field'):
            self.canvas.set_selected_field(field)
            
        # Sync control panel selection
        if hasattr(self.control_panel, 'set_selected_field'):
            self.control_panel.set_selected_field(field)
    
    def handle_box_complete(self, box: BoundingBox):
        if not self.current_image:
            logger.warning("No image loaded, cannot add annotation")
            return
            
        try:
            # Extract text from the box area
            extracted_text = OCRHandler.extract_text(
                self.current_image.current_page.image,
                (box.x1, box.y1, box.x2, box.y2)
            )
            
            field = self.control_panel.get_current_field()
            # Show dialog with pre-filled OCR text
            text = TextInputDialog.get_text(self, field, initial_text=extracted_text)
            
            if text:
                logger.debug(f"Adding annotation: Field={field}, Text={text}, Page={self.current_image.current_page_idx}")
                self.annotation_store.add_annotation(
                    field=field,
                    box=box,
                    text=text, 
                    page=self.current_image.current_page_idx
                )
                
                # Verify annotation was stored
                current_annotations = self.annotation_store.get_page_annotations(
                    self.current_image.current_page_idx
                )
                logger.debug(f"Current page annotations after add: {current_annotations}")
                
                self.canvas.update_annotations(current_annotations)
                self.control_panel.update_annotation_list(current_annotations)
                
        except Exception as e:
            logger.error(f"Error adding annotation: {e}")
            messagebox.showerror(
                "Annotation Error",
                f"Failed to add annotation: {str(e)}"
            )

    def handle_field_change(self, new_field: str) -> None:
        """Handle when user changes the selected field."""
        try:
            # Update canvas's current field
            self.canvas.set_current_field(new_field)
            
            # If we have a current image, refresh the annotations
            if self.current_image:
                current_page_annotations = self.annotation_store.get_page_annotations(
                    self.current_image.current_page_idx
                )
                self.canvas.update_annotations(current_page_annotations)
        except Exception as e:
            logger.error(f"Field change error: {str(e)}")
            messagebox.showerror(
                "Field Change Error",
                f"Error updating field selection: {str(e)}"
            )
    
    def load_image(self):
        try:
            path = filedialog.askopenfilename(
                filetypes=[
                    ("All supported", "*.pdf *.tiff *.tif *.png *.jpg *.jpeg"),
                    ("PDF files", "*.pdf"),
                    ("TIFF files", "*.tiff *.tif"),
                    ("Image files", "*.png *.jpg *.jpeg")
                ])
            if path:
                logger.debug(f"Loading image from path: {path}")
                self.current_image = ImageData(path)
                
                # Load existing annotations if they exist
                loaded_store = FileHandler.load_annotations(path)
                if loaded_store is not None:
                    logger.debug("Found existing annotations, loading them")
                    self.annotation_store = loaded_store
                else:
                    logger.debug("No existing annotations found, starting fresh")
                    self.annotation_store = AnnotationStore()
                
                # Update display
                self.update_display()
                
                # Update annotation list in control panel with current page annotations
                current_annotations = self.annotation_store.get_page_annotations(
                    self.current_image.current_page_idx
                )
                self.control_panel.update_annotation_list(current_annotations)
                
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            messagebox.showerror(
                "Error Loading Image",
                f"Failed to load image: {str(e)}"
            )
    
    def update_display(self):
        if not self.current_image or not self.current_image.current_page:
            return
            
        try:
            logger.debug("Updating display")
            current_page = self.current_image.current_page
            self.photo = ImageUtils.create_photo_image(current_page.image)
            
            self.canvas.config(
                width=current_page.image.width,
                height=current_page.image.height
            )
            self.canvas.delete("all")  # Clear previous content
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            
            # Get and display current page annotations
            current_annotations = self.annotation_store.get_page_annotations(
                self.current_image.current_page_idx
            )
            logger.debug(f"Current page annotations: {current_annotations}")
            self.canvas.update_annotations(current_annotations)
            
            # Update page display
            self.control_panel.update_page_display(
                self.current_image.current_page_idx + 1,
                self.current_image.total_pages
            )
            
            # Update annotation list
            self.control_panel.update_annotation_list(current_annotations)
            
        except Exception as e:
            logger.error(f"Display update error: {str(e)}")
            messagebox.showerror(
                "Display Error",
                f"Error updating display: {str(e)}"
            )
    
    def next_page(self):
        if self.current_image and self.current_image.next_page():
            logger.debug("Moving to next page")
            self.update_display()
    
    def prev_page(self):
        if self.current_image and self.current_image.prev_page():
            logger.debug("Moving to previous page")
            self.update_display()
    
    def save_annotations(self):
        if not self.current_image:
            logger.warning("No image loaded, cannot save annotations")
            messagebox.showwarning(
                "No Image",
                "Please load an image before saving annotations."
            )
            return
            
        try:
            logger.debug(f"Current annotation store before saving: {self.annotation_store.annotations}")
            FileHandler.save_annotations(
                self.current_image.path,
                self.annotation_store
            )
            logger.info("Annotations saved successfully")
            messagebox.showinfo(
                "Success",
                "Annotations saved successfully!"
            )
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
            messagebox.showerror(
                "Save Error",
                f"Failed to save annotations: {str(e)}"
            )