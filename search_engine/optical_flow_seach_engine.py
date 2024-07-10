import cv2
import numpy as np

from app_logger import get_logger
from .base_search_engine import BaseSearchEngine

logger = get_logger(__name__)

class SearchEngineOpticalFlow(BaseSearchEngine):
    def __init__(self, threshold, **kwargs) -> None:
        super().__init__(threshold, **kwargs)
        self._parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self._colours = np.random.randint(0, 255, (100, 3))
        self._initial_frame = True
        self._parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
        self._i = 0

    def run(self, obj):
        return_value = None
        img = obj.img
        if self._initial_frame:
            self._frame_gray_init = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._initial_frame = False
            self._edges = cv2.goodFeaturesToTrack(self._frame_gray_init, mask = None, **self._parameters_shitomasi)
            self._canvas = np.zeros_like(img)
            return obj
        
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       

        # update object corners by comparing with found edges in initial frame
        update_edges, status, errors = cv2.calcOpticalFlowPyrLK(self._frame_gray_init, frame_gray, self._edges, None,
                                                            **self._parameter_lucas_kanade)
        # only update edges if algorithm successfully tracked
        new_edges = update_edges[status == 1]
        # to calculate directional flow we need to compare with previous position
        old_edges = self._edges[status == 1]
        if len(update_edges[status == 0]) > 0:
            logger.info(f'search engine len {len(update_edges[status == 0])}')
            return_value = obj
        
        for new_e, old_e in zip(new_edges, old_edges):
            a, b = new_e.ravel()
            c, d = old_e.ravel()
            distance = (a-c) ** 2 + (b - d) ** 2
            if distance > self._threshold:
                logger.info(f'search engine distance {distance}')
                return_value = obj
                break        
        # overwrite initial frame with current before restarting the loop
        self._frame_gray_init = frame_gray.copy()
        # update to new edges before restarting the loop
        self._edges = new_edges.reshape(-1, 1, 2)
        return return_value

    def run_batch(self, images_with_name):
        unique_images_w_n = []
        for i, img_w_n in enumerate(images_with_name):
            res = self.run(img_w_n)
            if res:
                unique_images_w_n.append(img_w_n)
        return unique_images_w_n