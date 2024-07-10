import cv2

from .base_reader import BaseReader
from image_with_name_dc import ImageWithName


class OpenCVVideoReader(BaseReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        video = kwargs.pop('video')
        self._vidcap = cv2.VideoCapture(video)
        

    def read_data(self, *args, **kwargs):
        while True:
            
            success, frame = self._vidcap.read()
            if not success:
                break
            yield ImageWithName(None, frame)
