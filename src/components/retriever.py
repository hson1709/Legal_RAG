from abc import ABC, abstractmethod
from typing import Union, List

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: Union[str, List[str]]) -> str:
        pass
