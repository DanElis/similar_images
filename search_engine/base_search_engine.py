from abc import ABC, abstractclassmethod


class BaseSearchEngine(ABC):
    def __init__(self, threshold, **kwargs) -> None:
        self._threshold = threshold
    
    @abstractclassmethod
    def run(self, obj):
        raise NotImplementedError
    
    @abstractclassmethod
    def run_batch(self, images):
        raise NotImplementedError
