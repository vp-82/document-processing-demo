# models/image_data.py
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
from config.config import Config


@dataclass
class PageData:
    image: Image.Image
    page_number: int


class ImageData:
    def __init__(self, path: str):
        self.path = path
        self.pages: List[PageData] = []
        self.current_page_idx = 0
        self._load_pages()

    def _load_pages(self):
        # Open PDF or multi-page TIFF
        images = Image.open(self.path)
        try:
            for i in range(100):  # Safety limit of 100 pages
                try:
                    images.seek(i)
                    # Convert to RGB if necessary (for PDF pages)
                    if images.mode != "RGB":
                        page_image = images.convert("RGB")
                    else:
                        page_image = images.copy()

                    # Resize if needed
                    page_image.thumbnail(
                        Config.DEFAULT_DISPLAY_SIZE, Image.Resampling.LANCZOS
                    )
                    self.pages.append(PageData(page_image, i + 1))
                except EOFError:
                    break  # No more pages
        except Exception as e:
            print(f"Error loading pages: {e}")
            # If multi-page loading fails, try loading as single image
            if not self.pages:
                image = Image.open(self.path)
                image.thumbnail(Config.DEFAULT_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                self.pages.append(PageData(image, 1))

    @property
    def current_page(self) -> Optional[PageData]:
        if not self.pages:
            return None
        return self.pages[self.current_page_idx]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def next_page(self) -> bool:
        if self.current_page_idx < len(self.pages) - 1:
            self.current_page_idx += 1
            return True
        return False

    def prev_page(self) -> bool:
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            return True
        return False

    def goto_page(self, page_num: int) -> bool:
        if 0 <= page_num < len(self.pages):
            self.current_page_idx = page_num
            return True
        return False
