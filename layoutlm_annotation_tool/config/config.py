# config/config.py
from typing import Dict
import tkinter as tk

class Config:
    DEFAULT_WINDOW_TITLE = "Invoice Annotator"
    DEFAULT_DISPLAY_SIZE = (800, 1000)
    DEFAULT_FIELDS = [
        'total_amount',
        'vendor_name',
        'invoice_date',
        'invoice_number',
        'address'
    ]
    
    # Define colors for different fields
    FIELD_COLORS: Dict[str, str] = {
        'total_amount': '#FF6B6B',     # Red
        'vendor_name': '#4ECDC4',      # Teal
        'invoice_date': '#45B7D1',     # Blue
        'invoice_number': '#96CEB4',    # Green
        'address': '#FFEEAD'           # Yellow
    }
    
    # Box styling
    ACTIVE_BOX_WIDTH = 2
    INACTIVE_BOX_WIDTH = 1
    ACTIVE_BOX_OPACITY = 1.0
    INACTIVE_BOX_OPACITY = 0.5
    
    # Font configuration for labels
    LABEL_FONT = ('Arial', 10)
    LABEL_BG = '#FFFFFF'
    LABEL_PADDING = 2