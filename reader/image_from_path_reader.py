from abc import ABC, abstractclassmethod

import os
import cv2
import numpy as np
from PIL import Image
from torchvision.io.image import read_image

from .base_reader import BaseReader
import app_logger as app_logger
from image_with_name_dc import ImageWithName
from concurrent.futures import ThreadPoolExecutor


logger = app_logger.get_logger(__name__)

class ImageFromPathReader(BaseReader, ABC):
    def __init__(self, **kwargs):
        self._path_dir = kwargs.pop("path_dir")
        super().__init__(**kwargs)
        self._image_names = os.listdir(self._path_dir) 
        # self._num_threads = kwargs.pop('num_threads')
        # self._pool = ThreadPoolExecutor(self._num_threads)

    def read_data(self, *args, **kwargs):
        
        for im_n in self._image_names:
            if not self._is_img(im_n):
                continue
            img = self._read_img(os.path.join(self._path_dir, im_n))
            yield ImageWithName(im_n, img)
    
    def _is_img(self, image_name) -> bool:
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            return True
        else:
            return False
    
    @abstractclassmethod
    def _read_img(self, path_img):
        raise NotImplementedError


class PILImageFromPathReader(ImageFromPathReader):
    def __init__(self, **kwargs):
        self._convert_to_np = kwargs.pop('convert_to_np')
        super().__init__(**kwargs)

    def _read_img(self, path_img):
        img = Image.open(path_img)
        if self._convert_to_np:
            img = np.array(img)
        return img
    
class OpenCVImageFromPathReader(ImageFromPathReader):
    def __init__(self, **kwargs):
        self._convert_to_rgb = kwargs.pop('convert_to_rgb')
        super().__init__(**kwargs)

    def _read_img(self, path_img):
        img = cv2.imread(path_img)
        if self._convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class TorchvisionIOFromPathReader(ImageFromPathReader):
    def _read_img(self, path_img):
        img = read_image(path_img)
        return img
    
class OpenCVImageFromPathReaderWithCustomSorted(OpenCVImageFromPathReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._image_names = sorted(self._image_names, key=lambda x: int(x[6:-4]))

class PILImageFromPathReaderWithCustomSorted(PILImageFromPathReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._image_names = sorted(self._image_names, key=lambda x: int(x[6:-4]))

class TorchvisionIOhReaderWithCustomSorted(TorchvisionIOFromPathReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._image_names = sorted(self._image_names, key=lambda x: int(x[6:-4]))