from typing_extensions import List
from abc import ABC, abstractmethod
from src.domain.entity.Deteccion import Deteccion

class IImageProcessor(ABC):
    
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def predict(self, frame: object, stream: bool = True) -> List[Deteccion]:
        pass