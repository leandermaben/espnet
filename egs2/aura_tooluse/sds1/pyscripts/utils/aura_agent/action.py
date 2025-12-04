"""Base action class for Aura agent."""


class Action:
    """Base class for all agent actions.

    Actions represent different capabilities that the agent can perform,
    such as web search, calendar management, or direct chat responses.
    """

    def __init__(self, thought: str = "", payload: str = ""):
        """Initialize an action.

        Args:
            thought: The agent's reasoning for taking this action.
            payload: The data or query associated with the action.
        """
        self.thought = thought
        self.payload = payload

    def execute(self, state):
        """Execute the action and update the state.

        This method should be overridden by subclasses to implement
        specific action behavior.

        Args:
            state: The current conversation state.

        Returns:
            observation: The result of executing the action.
        """
        raise NotImplementedError("Subclasses must implement execute method")


class ChatAction(Action):
    """Action for direct chat responses without tool use."""

    def execute(self, state):
        """Execute chat action by returning the response.

        Args:
            state: The current conversation state.

        Returns:
            str: The chat response payload.
        """
        # Append to conversation history
        state.add_message("assistant", self.payload)
        return self.payload


class WebSearchAction(Action):
    """Action for performing web searches."""

    def execute(self, state):
        """Execute web search and return results.

        Args:
            state: The current conversation state.

        Returns:
            str: Formatted search results.
        """
        try:
            import requests

            # Using DuckDuckGo Instant Answer API (no API key required)
            query = self.payload
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            # Format the results
            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

            # Add related topics
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(f"- {topic['Text']}")

            observation = (
                "\n".join(results) if results else "No detailed results found."
            )

        except Exception as e:
            observation = f"Search failed: {str(e)}"

        # Append to history
        state.add_observation(f"Search query: {query}", observation)
        return observation


class CalculatorAction(Action):
    """Action for performing calculations."""

    def execute(self, state):
        """Execute calculation and return result.

        Args:
            state: The current conversation state.

        Returns:
            str: The calculation result.
        """
        try:
            # Safely evaluate mathematical expressions
            # Remove any non-mathematical characters for safety
            expression = self.payload
            allowed_chars = set("0123456789+-*/().% ")
            cleaned = "".join(c for c in expression if c in allowed_chars)

            # Use eval with restricted namespace for safety
            result = eval(
                cleaned, {"__builtins__": {}}, {"abs": abs, "round": round, "pow": pow}
            )
            observation = f"Result: {result}"
        except Exception as e:
            observation = f"Calculation failed: {str(e)}"

        state.add_observation(f"Calculate: {self.payload}", observation)
        return observation
