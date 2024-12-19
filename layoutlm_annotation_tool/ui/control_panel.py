# ui/control_panel.py
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional
import logging

from models.annotation import Annotation
from config.config import Config

logger = logging.getLogger(__name__)

class ControlPanel(tk.Frame):
    def __init__(
        self,
        master,
        on_load_image: Callable[[], None],
        on_save: Callable[[], None],
        on_prev_page: Callable[[], None],
        on_next_page: Callable[[], None],
        on_field_change: Callable[[str], None],
        on_selection_change: Callable[[Optional[str]], None],
        on_edit_annotation: Callable[[str], None],
        on_delete_annotation: Callable[[str], None]
    ):
        super().__init__(master)
        self.on_field_change = on_field_change
        self.on_selection_change = on_selection_change
        self.on_edit_annotation = on_edit_annotation
        self.on_delete_annotation = on_delete_annotation
        
        # Store the currently selected annotation
        self.selected_field: Optional[str] = None
        
        # Load image button
        tk.Button(self, text="Load Image",
                 command=on_load_image).pack(pady=5)
        
        # Page navigation
        self.page_frame = tk.Frame(self)
        self.page_frame.pack(pady=5)
        tk.Button(self.page_frame, text="◀", command=on_prev_page).pack(side=tk.LEFT)
        self.page_label = tk.Label(self.page_frame, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=5)
        tk.Button(self.page_frame, text="▶", command=on_next_page).pack(side=tk.LEFT)
        
        # Field selection with color indicators
        tk.Label(self, text="Fields:").pack(pady=5)
        self.field_var = tk.StringVar(value=Config.DEFAULT_FIELDS[0])
        
        # Create field buttons with color indicators
        self.field_buttons: Dict[str, tk.Radiobutton] = {}
        for field in Config.DEFAULT_FIELDS:
            frame = tk.Frame(self)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Color indicator
            color_indicator = tk.Frame(
                frame,
                width=10,
                height=10,
                bg=Config.FIELD_COLORS[field]
            )
            color_indicator.pack(side=tk.LEFT, padx=(0, 5))
            
            # Radio button
            btn = tk.Radiobutton(
                frame,
                text=field,
                variable=self.field_var,
                value=field,
                command=self._on_field_selected
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.field_buttons[field] = btn
        
        # Save button
        tk.Button(self, text="Save Annotations",
                 command=on_save).pack(pady=20)
        
        # Annotation list with selection support
        tk.Label(self, text="Annotations:").pack(pady=(5, 0))
        self.annotation_list = tk.Listbox(self, width=40, height=10)
        self.annotation_list.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.annotation_list.bind('<<ListboxSelect>>', self._handle_list_selection)
        self.annotation_list.bind("<Button-3>", self.show_context_menu)
        self.annotation_list.bind("<Delete>", lambda e: self.handle_delete())
        self.annotation_list.bind("<Return>", lambda e: self.handle_edit())
        
        logger.debug("Initialized ControlPanel with selection support")

    def show_context_menu(self, event):
        """Show context menu for annotation editing."""
        if not self.selected_field:
            return
            
        menu = Menu(self, tearoff=0)
        menu.add_command(label="Edit (Enter)", 
                        command=self.handle_edit)
        menu.add_command(label="Delete (Del)", 
                        command=self.handle_delete)
        menu.post(event.x_root, event.y_root)

    def handle_delete(self):
        """Handle delete command from menu or keyboard."""
        if self.selected_field:
            self.on_delete_annotation(self.selected_field)

    def handle_edit(self):
        """Handle edit command from menu or keyboard."""
        if self.selected_field:
            self.on_edit_annotation(self.selected_field)
    
    def _handle_list_selection(self, event):
        """Handle selection change in the annotation list."""
        selection = self.annotation_list.curselection()
        if not selection:
            self.selected_field = None
        else:
            idx = selection[0]
            item_text = self.annotation_list.get(idx)
            # Extract field from the list item text
            self.selected_field = item_text.split(':')[0].strip()
            
        logger.debug(f"List selection changed to: {self.selected_field}")
        self.on_selection_change(self.selected_field)
    
    def set_selected_field(self, field: Optional[str]) -> None:
        """Update the list selection when canvas selection changes."""
        if field == self.selected_field:
            return
            
        self.selected_field = field
        
        # Update listbox selection
        self.annotation_list.selection_clear(0, tk.END)
        if field:
            # Find and select the corresponding item
            for i in range(self.annotation_list.size()):
                if self.annotation_list.get(i).startswith(field + ':'):
                    self.annotation_list.selection_set(i)
                    self.annotation_list.see(i)  # Ensure the item is visible
                    break
    
    def _on_field_selected(self):
        """Handle field selection change."""
        self.on_field_change(self.field_var.get())
    
    def get_current_field(self) -> str:
        """Get the currently selected field."""
        return self.field_var.get()
    
    def update_page_display(self, current: int, total: int) -> None:
        """Update the page number display."""
        self.page_label.config(text=f"Page: {current}/{total}")
    
    def update_annotation_list(self, annotations: Dict[str, Annotation]) -> None:
        """Update the list of annotations with selection preservation."""
        self.annotation_list.delete(0, tk.END)
        
        for field, annotation in annotations.items():
            # Format the text with field and value
            text = f"{field}: {annotation.text}"
            self.annotation_list.insert(tk.END, text)
            
        # Restore selection if applicable
        if self.selected_field:
            self.set_selected_field(self.selected_field)