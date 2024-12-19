# main.py
import sys
import tkinter as tk
# from pathlib import Path

# Add the project root to the Python path
# project_root = Path(__file__).parent
# sys.path.append(str(project_root))

from ui.main_window import MainWindow
# from config.config import Config


def main():
    """
    Main entry point for the LayoutLM Annotation Tool.
    Initializes and runs the main application window.
    """
    try:
        app = MainWindow()
        # Center the window on screen
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()
        window_width = 1200  # Default width
        window_height = 800  # Default height
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        app.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        app.mainloop()
    except Exception as e:
        tk.messagebox.showerror(
            "Error",
            f"An error occurred while starting the application:\n{str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()