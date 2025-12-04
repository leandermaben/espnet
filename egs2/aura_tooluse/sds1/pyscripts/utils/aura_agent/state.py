"""State management for Aura agent."""


class State:
    """Manages conversation state and history for the agent."""

    def __init__(self):
        """Initialize conversation state."""
        self.messages = []  # List of (role, content) tuples
        self.observations = []  # List of (action, observation) tuples

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: The role of the message sender ('user' or 'assistant').
            content: The message content.
        """
        self.messages.append({"role": role, "content": content})

    def add_observation(self, action: str, observation: str):
        """Add an action-observation pair to the history.

        Args:
            action: Description of the action taken.
            observation: The result of the action.
        """
        self.observations.append({"action": action, "observation": observation})

    def get_conversation_history(self) -> list:
        """Get the full conversation history.

        Returns:
            list: List of message dictionaries.
        """
        return self.messages

    def get_last_user_message(self) -> str:
        """Get the most recent user message.

        Returns:
            str: The last user message content, or empty string if none.
        """
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def format_for_llm(self) -> list:
        """Format conversation history for LLM consumption.

        Returns:
            list: Formatted messages for LLM API.
        """
        formatted = []
        for msg in self.messages:
            formatted.append({"role": msg["role"], "content": msg["content"]})

        # Include recent observations in the context
        if self.observations:
            obs_text = "\n\nRecent tool observations:\n"
            for obs in self.observations[-3:]:  # Last 3 observations
                obs_text += f"- {obs['action']}: {obs['observation']}\n"
            if formatted:
                # Add observations to the last assistant message or create new one
                formatted.append(
                    {"role": "system", "content": obs_text.strip()}
                )

        return formatted

    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        self.observations = []
