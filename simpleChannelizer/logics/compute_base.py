import abc
import numpy as np
from pydantic import BaseModel


class ComputeBase(abc.ABC):
    class Config(BaseModel):
        """Base config for all computing logics"""

        def create_logical_instance(self) -> "ComputeBase":
            pass
    def __init__(self, config: "ComputeBase.Config"):
        self.config = config
        self.initialize()

    def initialize(self):
        pass

    @abc.abstractmethod
    def compute(self, data: np.ndarray) -> np.ndarray:
        pass