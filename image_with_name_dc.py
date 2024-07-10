from typing import Union, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

@dataclass
class ImageWithName:
    def __init__(self, name, img):
        self.name = name
        self.img = img
