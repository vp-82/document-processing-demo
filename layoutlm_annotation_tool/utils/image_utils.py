# utils/image_utils.py
from PIL import Image, ImageTk


class ImageUtils:
    @staticmethod
    def create_photo_image(image: Image.Image) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(image)
