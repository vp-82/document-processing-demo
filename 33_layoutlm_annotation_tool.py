import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import os

class InvoiceAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Invoice Annotator")
        
        # Data storage
        self.current_image_path = None
        self.annotations = {}
        self.current_box = None
        self.drawing = False
        
        # Fields we want to annotate
        self.fields = [
            'total_amount',
            'vendor_name',
            'invoice_date',
            'invoice_number',
            'address'
        ]
        self.current_field = self.fields[0]
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_box)
        
        # Controls
        tk.Button(self.control_frame, text="Load Image", 
                 command=self.load_image).pack(pady=5)
        
        # Field selection
        tk.Label(self.control_frame, text="Current Field:").pack(pady=5)
        self.field_var = tk.StringVar(value=self.fields[0])
        for field in self.fields:
            tk.Radiobutton(self.control_frame, text=field, 
                          variable=self.field_var, value=field).pack()
        
        # Save button
        tk.Button(self.control_frame, text="Save Annotations", 
                 command=self.save_annotations).pack(pady=20)
        
        # Annotation list
        self.annotation_list = tk.Listbox(self.control_frame, width=40, height=10)
        self.annotation_list.pack(pady=5)
        
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if path:
            self.current_image_path = path
            self.load_current_image()
            
    def load_current_image(self):
        # Load and display image
        image = Image.open(self.current_image_path)
        # Resize if too large
        display_size = (800, 1000)  # Maximum display size
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        # Clear existing annotations
        self.annotations = {}
        self.update_annotation_list()
        
    def start_box(self, event):
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        if self.current_box:
            self.canvas.delete(self.current_box)
        
    def draw_box(self, event):
        if self.drawing:
            if self.current_box:
                self.canvas.delete(self.current_box)
            
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            
            self.current_box = self.canvas.create_rectangle(
                self.start_x, self.start_y, cur_x, cur_y,
                outline='red'
            )
            
    def end_box(self, event):
        self.drawing = False
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Save annotation
        field = self.field_var.get()
        coords = [
            min(self.start_x, end_x),
            min(self.start_y, end_y),
            max(self.start_x, end_x),
            max(self.start_y, end_y)
        ]
        
        # Get text content for this box
        text = self.get_text_input(field)
        if text:
            self.annotations[field] = {
                'box': coords,
                'text': text
            }
            self.update_annotation_list()
            
    def get_text_input(self, field):
        # Simple dialog for text input
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Enter {field}")
        
        text_var = tk.StringVar()
        tk.Entry(dialog, textvariable=text_var).pack(pady=5)
        
        result = [None]  # Use list to store result
        
        def save():
            result[0] = text_var.get()
            dialog.destroy()
            
        tk.Button(dialog, text="Save", command=save).pack(pady=5)
        
        dialog.wait_window()  # Wait for dialog to close
        return result[0]
    
    def update_annotation_list(self):
        self.annotation_list.delete(0, tk.END)
        for field, data in self.annotations.items():
            self.annotation_list.insert(tk.END, 
                f"{field}: {data['text']} {data['box']}")
    
    def save_annotations(self):
        if not self.current_image_path:
            return
            
        # Create annotation file path
        base_path = os.path.splitext(self.current_image_path)[0]
        json_path = f"{base_path}_annotations.json"
        
        # Save annotations
        data = {
            'image_path': self.current_image_path,
            'annotations': self.annotations
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved annotations to {json_path}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = InvoiceAnnotator(root)
    root.mainloop()