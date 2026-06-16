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

    @abstractmethod
    def set_context(self, context: dict[str, Any] | None):
        """
        Set context from previous task for follow-up queries.
        
        Context includes:
        - last_screenshot_base64: Screenshot when previous task completed
        - last_url: URL when previous task completed
        - last_thought: Agent's last thought
        - conversation_history: List of previous queries/actions
        """
        pass