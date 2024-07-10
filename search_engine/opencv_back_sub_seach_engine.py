import cv2

from .base_search_engine import BaseSearchEngine
from app_logger import get_logger

logger = get_logger(__name__)

class SearchEngineOpenCVBackgroundSubtractor(BaseSearchEngine):
    def __init__(self, threshold, **kwargs):
        self._type_background_substractor = kwargs.pop("type_background_substractor")
        super().__init__(threshold, **kwargs)
        self._back_sub = self._init_background_substractor(self._type_background_substractor)
    
    def _init_background_substractor(self, type_background_substractor):
        if type_background_substractor == 'mog2':
            back_sub = cv2.createBackgroundSubtractorMOG2()
        elif type_background_substractor == 'knn':
            back_sub = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError(f'type_background_substractor must be mog2 or knn. Get {type_background_substractor}')
        return back_sub

    def _is_same(self, fs_m):
        h, w = fs_m.shape
        img_area = h * w
        contours, _ = cv2.findContours(fs_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum_contours = 0
        for contour in contours:
            sum_contours += cv2.contourArea(contour)
        logger.debug(f'is_same {sum_contours / img_area}')
        if sum_contours / img_area > self._threshold:
            return False
        return True

    def run(self, obj):
        img = obj.img
        fs_m = self._back_sub.apply(img) 
        if not self._is_same(fs_m):
            return obj
        return None
    
    def run_batch(self, images_with_name):
        unique_images_w_n = []
        for i, img_w_n in enumerate(images_with_name):
            res = self.run(img_w_n)
            if res:
                unique_images_w_n.append(img_w_n)
        return unique_images_w_n
