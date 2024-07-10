from abc import ABC, abstractclassmethod


class BaseReader(ABC):

    @abstractclassmethod
    def read_data(self, *args, **kwargs):
        raise NotImplementedError
