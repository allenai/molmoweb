import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agent.actions import ALL_ACTIONS, SendMsgToUser
from agent.utils import AgentBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Interaction:
    state: dict[str, Any]
    action: dict[str, Any] | None = None
    raw_output: str | None = None
    next_state: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    action_timestamp: float | None = None


class Episode:
    def __init__(
        self, env, agent: AgentBase, eps_name: str, goal: str | None = None
    ):
        self.env = env
        self.agent = agent
        self.eps_name = eps_name
        self.goal = goal
        self.interactions: list[Interaction] = []
        self.metadata: dict[str, Any] = {}

    def _predict_with_retry(self, obs: dict):
        result = self.agent.predict_action(obs)

        if isinstance(result, tuple) and len(result) == 2:
            pred_text, action = result
            if pred_text is None:
                raise ValueError(f"Trying again because pred_text is None")
            if pred_text and pred_text.startswith("Predictor error:"):
                raise ValueError(f"Trying again because: {pred_text}")
        else:
            action = result

        if not isinstance(action, dict) or "action_str" not in action:
            raise ValueError("Invalid action: missing 'action_str'")

        return result

    def _env_step(self, action_obj: ALL_ACTIONS, action_str: str | None = None):
        """Execute an action on the environment."""
        from utils.envs.browser_env import BrowserEnv

        if isinstance(self.env, BrowserEnv):
            return self.env.step(action=action_obj), None, None, None, None

        return self.env.step(action_str)

    def run_episode(self, max_steps: int = 30):
        obs, info = self.env.reset()
        self.agent.reset()
        if self.goal is None:
            self.goal = obs.get("goal")

        state = {
            "obs": obs,
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "info": info,
        }
        self.metadata = {"goal": self.goal, "eps_name": self.eps_name}

        for i in range(max_steps):
            action_timestamp = datetime.now(timezone.utc).timestamp()

            try:
                result = self._predict_with_retry(state["obs"])
                raw_output, action = (
                    result
                    if isinstance(result, tuple) and len(result) == 2
                    else (None, result)
                )
                raw_output = "" if raw_output is None else raw_output

                state["agent_inputs"] = getattr(
                    self.agent, "get_last_model_inputs", lambda: None
                )()
            except Exception as e:
                error = {"ACTION_PREDICTION_ERROR": str(e)}
                self.interactions.append(Interaction(state=state, error=error))
                return self.interactions, self.metadata

            try:
                action_obj = action["action_output"].action
                obs, reward, terminated, truncated, info = self._env_step(
                    action_obj=action_obj,
                    action_str=action.get("action_str"),
                )
                next_state = {
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            except Exception as e:
                error = {"ENV_STEP_ERROR": str(e)}
                self.interactions.append(
                    Interaction(
                        state=state,
                        action=action,
                        raw_output=raw_output,
                        action_timestamp=action_timestamp,
                        error=error,
                    )
                )
                return self.interactions, self.metadata

            self.interactions.append(
                Interaction(
                    state=state,
                    action=action,
                    raw_output=raw_output,
                    next_state=next_state,
                    action_timestamp=action_timestamp,
                )
            )

            state = next_state
            if terminated or (
                isinstance(action_obj, SendMsgToUser)
                and action_obj.msg
                and action_obj.msg.startswith("[EXIT]")
            ):
                print("Episode ended by agent action or termination signal.")
                break

        return self.interactions, self.metadata
