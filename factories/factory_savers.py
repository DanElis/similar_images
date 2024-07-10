from .base_factory import BaseFactory
from saver import BaseSaver, PILSaver, OpenCVSaver
from singleton import Singleton


class FactorySaver(BaseFactory):
    def __init__(self) -> None:
        self._name2class = {
            'pil': PILSaver,
            'opencv': OpenCVSaver,
        }

    def __call__(self, *args, **kwargs) -> BaseSaver:
        name = kwargs.pop('name')
        try:
            object_creator = self._name2class[name]
        except KeyError as ex:
            raise ValueError(f"Name Saver must be {self._name2class.keys()}. Get {name}")
        return object_creator(*args, **kwargs)