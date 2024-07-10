from .base_factory import BaseFactory
from reader import (
    BaseReader, 
    OpenCVVideoReader, 
    PILImageFromPathReader, 
    OpenCVImageFromPathReader, 
    TorchvisionIOFromPathReader, 
    OpenCVVideoReaderRetryConnect,
    PILImageFromPathReaderWithCustomSorted,
    OpenCVImageFromPathReaderWithCustomSorted,
    TorchvisionIOhReaderWithCustomSorted,
)
from singleton import Singleton


class FactoryReaders(BaseFactory):
    def __init__(self) -> None:
        self._name2class = {
            'pil-path': PILImageFromPathReader,
            'opencv-path': OpenCVImageFromPathReader,
            'opencv-video': OpenCVVideoReader,
            'torchvisionIO': TorchvisionIOFromPathReader,
            'opencv-video-retry-connect': OpenCVVideoReaderRetryConnect,
            'pil-path': PILImageFromPathReader,
            'opencv-path': OpenCVImageFromPathReader,
            'pil-path-custom-sorted': PILImageFromPathReaderWithCustomSorted,
            'opencv-path-custom-sorted': OpenCVImageFromPathReaderWithCustomSorted,
            'torchvisionIO-custom-sorted': TorchvisionIOhReaderWithCustomSorted,
        }

    def __call__(self, *args, **kwargs) -> BaseReader:
        name = kwargs.pop('name')
        try:
            object_creator = self._name2class[name]
        except KeyError as ex:
            raise ValueError(f"Name Reader must be {self._name2class.keys()}. Get {name}")
        return object_creator(*args, **kwargs)