from abc import ABC, abstractclassmethod
from typing import Any


class BaseFactory(ABC):
    @abstractclassmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError
