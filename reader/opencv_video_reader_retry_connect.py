import cv2

from .base_reader import BaseReader
from image_with_name_dc import ImageWithName
from app_logger import get_logger

logger = get_logger(__name__)

class OpenCVVideoReaderRetryConnect(BaseReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._video = kwargs.pop('video')
        self._vidcap = self._try_connect_to_stream()

    def _try_connect_to_stream(self):
        logger.info(f'try_connect_to_stream {self._video}')
        vidcap = cv2.VideoCapture(self._video)
        return vidcap

    def read_data(self, *args, **kwargs):
        while True:
            
            success, frame = self._vidcap.read()
            if not success:
                self._vidcap = self._try_connect_to_stream()
            yield ImageWithName(None, frame)
