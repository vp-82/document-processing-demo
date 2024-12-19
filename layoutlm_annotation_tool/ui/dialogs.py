# ui/dialogs.py
import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger(__name__)

class TextInputDialog:
    @staticmethod
    def get_text(parent, field: str, initial_text: str = "") -> str:
        """
        Show dialog for text input with initial OCR text.
        Converts newlines to separator in the final output.
        
        Args:
            parent: Parent window
            field: Field name
            initial_text: Pre-filled text (e.g., from OCR)
            
        Returns:
            str: Entered text with newlines replaced by separator, or empty string if cancelled
        """
        # Convert separator back to newlines for display
        display_text = initial_text.replace(" | ", "\n")
        
        dialog = tk.Toplevel(parent)
        dialog.title(f"Enter {field}")
        dialog.transient(parent)
        dialog.grab_set()
        
        # Make dialog larger for multiline text
        dialog.geometry("400x250")
        dialog.resizable(True, True)
        
        # Create and pack widgets
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Enter or verify {field}:").pack(pady=(0, 5))
        
        # Text widget for multiline input
        text_widget = tk.Text(frame, width=40, height=8, wrap=tk.WORD)
        text_widget.insert('1.0', display_text)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        text_widget.focus()
        
        result = {"text": ""}
        
        def on_ok():
            # Get text and convert newlines to separator
            text = text_widget.get('1.0', 'end-1c').strip()
            # Replace multiple newlines with single separator
            text = ' | '.join(line.strip() for line in text.splitlines() if line.strip())
            result["text"] = text
            dialog.destroy()
            
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        
        # Handle Return and Escape keys - modified for Text widget
        dialog.bind("<Control-Return>", lambda e: on_ok())  # Use Ctrl+Return since Return is for newlines
        dialog.bind("<Escape>", lambda e: on_cancel())
        
        # Wait for dialog to close
        parent.wait_window(dialog)
        
        return result["text"]