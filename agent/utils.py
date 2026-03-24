from abc import ABC, abstractmethod
from typing import Any


class AgentBase(ABC):
    @abstractmethod
    def reset(self):
        """
        Abstract method to reset agent's memory
        """
        pass

    @abstractmethod
    def predict_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Abstract method to predict the next action given the current observation
        """
        pass

    @abstractmethod
    def get_last_model_inputs(self) -> dict[str, Any] | None:
        """
        Abstract method to get the last model inputs
        """
        pass
