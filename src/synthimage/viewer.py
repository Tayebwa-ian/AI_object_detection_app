#!/usr/bin/python3"
"""
Utilities to display images. Works both in notebooks and normal Python.
"""

from PIL import Image
import sys
import os


class Viewer:
    """
    Simple display helper. In Jupyter, you get inline display; otherwise the default image viewer.
    """

    @staticmethod
    def display(img: Image.Image, title: str = None) -> None:
        """
        Show the image. In a notebook it will display inline; otherwise it opens default viewer.
        """
        try:
            # In notebook, IPython is available
            from IPython.display import display as ipy_display
            ipy_display(img)
        except Exception:
            # fallback to PIL's show (which opens the OS image viewer)
            if title:
                # when available, use temporary display name by saving with title as filename
                # but keep it simple: just call show()
                pass
            img.show()
