import os
from abc import ABC, abstractclassmethod
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import PIL
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
import cv2

class BaseSaver(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._ext = kwargs.pop('ext')
        self._make_dirs = kwargs.pop('make_dirs')
        self._path_to_save = kwargs.pop('path_to_save')
        self._i = 0
        super().__init__()
        
    async def save(self, obj, *args, **kwargs):
        name = obj.name
        img = obj.img
        if name is None:
            name = self._generate_name(self._ext)
        if self._make_dirs:
            os.makedirs(self._path_to_save, exist_ok=True)
        self._save_img(os.path.join(self._path_to_save, name), img)
    
    def _generate_name(self, ext):
        name = f'{self._i}.{ext}'
        self._i += 1
        return name
    
    @abstractclassmethod
    def _save_img(self, path, img):
        raise NotImplementedError 
    

class PILSaver(BaseSaver):

    def _convert_img(self, img):
        if isinstance(img, PIL.Image.Image):
            return img
        elif isinstance(img, np.ndarray):
            return Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            return to_pil_image(img)
        else:
            raise TypeError(f"img must be PIL.Image.Image, np.ndarray or torch.Tensor. Get {type(img)}")

    def _save_img(self, path, img):
        img = self._convert_img(img)
        img.save(path)

class OpenCVSaver(BaseSaver):

    def _convert_img(self, img):
        if isinstance(img, PIL.Image.Image):
            return np.array(img)
        elif isinstance(img, np.ndarray):
            return img
        elif isinstance(img, torch.Tensor):
            return img.cpu().data.numpy()
        else:
            raise TypeError(f"img must be PIL.Image.Image, np.ndarray or torch.Tensor. Get {type(img)}")

    def _save_img(self, path, img):
        img = self._convert_img(img)
        cv2.imwrite(path, img)
