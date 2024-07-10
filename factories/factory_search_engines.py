from .base_factory import BaseFactory
from search_engine import BaseSearchEngine, SearchEngineCLIP, SearchEngineTorchCam, SearchEngineOpenCVBackgroundSubtractor, SearchEngineOpticalFlow
from singleton import Singleton


class FactorySearchEngines(BaseFactory):
    def __init__(self) -> None:
        self._name2class = {
            'clip': SearchEngineCLIP,
            'torch-cam': SearchEngineTorchCam,
            'back-sub': SearchEngineOpenCVBackgroundSubtractor,
            'opticalflow': SearchEngineOpticalFlow,
        }

    def __call__(self, *args, **kwargs) -> BaseSearchEngine:
        name = kwargs.pop('name')
        try:
            object_creator = self._name2class[name]
        except KeyError as ex:
            raise ValueError(f"Name SearchEngine must be {self._name2class.keys()}. Get {name}")
        return object_creator(*args, **kwargs)
