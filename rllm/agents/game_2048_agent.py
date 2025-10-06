import copy
import logging
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


class Game2048Agent(BaseAgent):
    """Agent for playing the 2048 game."""

    SYSTEM_PROMPT: str = """You are an excellent 2048 player. Always choose the move most likely to combine tiles and eventually reach the target value. Valid moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>"""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool = True,
        use_accumulate_history: bool = True,
    ):
        """
        Initialize the Game2048Agent.

        Args:
            max_steps: Maximum number of steps allowed per episode
            use_accumulate_thinking: Whether to keep thinking in conversation history
            use_accumulate_history: Whether to accumulate full message history
        """
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step: int = 0
        self.accumulate_thinking: bool = use_accumulate_thinking
        self.max_steps: int | None = max_steps
        self.accumulate_history: bool = use_accumulate_history
        self.current_observation: Any = None
        self.reset()

    def update_from_env(
        self, observation: Any, reward: float, done: bool, info: dict, **kwargs
    ):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation: The current board state as a string
            reward: The reward received from the environment
            done: Whether the episode is finished
            info: Additional information from the environment
        """
        current_obs_str = str(observation)

        # Construct user prompt
        user_prompt_content = (
            f"Current Board (Step {self.step}):\n{current_obs_str}\n\n"
        )

        # Check if the last move was invalid (board didn't change)
        if (
            self._trajectory.steps
            and self._trajectory.steps[-1].action is not None
        ):
            last_step_obs_str = self._trajectory.steps[-1].observation
            if last_step_obs_str == current_obs_str:
                user_prompt_content += (
                    "WARNING: Your last move was invalid! The board didn't change.\n"
                    "Please ensure you:\n"
                    "1. Output a valid direction (Left, Right, Up, Down)\n"
                    "2. Format it correctly in ``` ```\n"
                    "3. Choose a move that will actually change the board state\n\n"
                )

        # Add status information
        if info.get("max_tile"):
            user_prompt_content += f"Current Max Tile: {info['max_tile']}\n"
        if info.get("target_value"):
            user_prompt_content += f"Target Value: {info['target_value']}\n"

        # Add remaining steps information
        if self.max_steps is not None and self.max_steps - self.step > 0:
            user_prompt_content += (
                f"Remaining Steps: {self.max_steps - self.step}\n"
            )

        user_prompt_content += "\nWhat is your next move?"

        self.messages.append({"role": "user", "content": user_prompt_content})
        self.current_observation = current_obs_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state after receiving a model response.

        Args:
            response: The model's text response

        Returns:
            Action object containing the parsed move
        """
        content = response

        # Optionally remove thinking tags
        if not self.accumulate_thinking:
            _, sep, after = content.partition("</think>")
            if sep:
                content = after

        thought, action_str = self._parse_model_response(content)

        # Create a new step in the trajectory
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=content,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        self.messages.append({"role": "assistant", "content": content})
        self.step += 1

        return Action(action=action_str)

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        """
        Parse the model's response to extract thought and action.
        Supports XML format: <move>left</move>

        Args:
            response: The raw model response

        Returns:
            Tuple of (thought, action_string)
        """
        import xml.etree.ElementTree as ET

        thought = response
        action_str = "invalid"

        # Try to parse XML format: <move>direction</move>
        try:
            # Find XML tags in the response
            xml_match = re.search(r"<move>(.*?)</move>", response, re.IGNORECASE)
            if xml_match:
                direction = xml_match.group(1).strip().lower()
                if direction in ["left", "right", "up", "down"]:
                    action_str = direction
                    # Everything before the XML tag is the thought
                    thought = response[:xml_match.start()].strip()
        except Exception:
            pass

        return thought, action_str

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Get the current chat completion messages.

        Returns:
            List of message dictionaries
        """
        if self.accumulate_history:
            return self.messages
        else:
            # Only keep system prompt and last user message
            if len(self.messages) <= 1:
                return self.messages
            else:
                return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        """Get the agent's trajectory."""
        return self._trajectory

    def reset(self) -> None:
        """Reset the agent's internal state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.step = 0
        self.current_observation = None

