"""Aura agent with ReAct-style tool use."""

import re

from .action import CalculatorAction, ChatAction, WebSearchAction
from .state import State


class AuraAgent:
    """Agent that can use tools via ReAct-style prompting."""

    SYSTEM_PROMPT = """You are Aura, a helpful voice assistant with access to tools.

You can help users with:
1. General conversation and questions
2. Web searches for current information
3. Simple calculations

When you need to use a tool, use this format:
Thought: [your reasoning about what to do]
Action: [tool name: web_search OR calculator OR chat]
Input: [the input for the tool]

Available tools:
- web_search: Search the web for information. Input should be a search query.
- calculator: Perform mathematical calculations. Input should be a math expression.
- chat: Direct response without using any tool. Input should be your response.

For most queries, you should respond directly using the 'chat' action.
Only use web_search when you need current information or facts you don't know.
Only use calculator for mathematical computations.

After using a tool (except chat), you'll see the result. Then provide a final response using the chat action.

Examples:

User: What's the weather like today?
Thought: I need to search for current weather information.
Action: web_search
Input: current weather today

User: What's 25 * 17?
Thought: I should use the calculator for this.
Action: calculator
Input: 25 * 17

User: Hello! How are you?
Thought: This is a simple greeting, I can respond directly.
Action: chat
Input: Hello! I'm doing well, thank you for asking! How can I help you today?
"""

    def __init__(self, llm_client):
        """Initialize the agent.

        Args:
            llm_client: The LLM client for generating responses.
        """
        self.llm_client = llm_client
        self.max_iterations = 3

    def parse_action(self, text: str):
        """Parse the LLM response to extract action.

        Args:
            text: The LLM response text.

        Returns:
            Action object or None if parsing fails.
        """
        # Extract thought, action, and input using regex
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n|Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Input:\s*(.+?)(?=\n\n|$)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action_type = action_match.group(1).strip().lower() if action_match else None
        action_input = input_match.group(1).strip() if input_match else ""

        # Map action type to action class
        if action_type == "web_search":
            return WebSearchAction(thought=thought, payload=action_input)
        elif action_type == "calculator":
            return CalculatorAction(thought=thought, payload=action_input)
        elif action_type == "chat":
            return ChatAction(thought=thought, payload=action_input)
        else:
            # Default to chat if we can't parse properly
            # Use the full text as the response
            return ChatAction(thought="Direct response", payload=text)

    def step(self, state: State):
        """Execute one step of the agent.

        Args:
            state: The current conversation state.

        Returns:
            tuple: (action, observation) pair.
        """
        # Build messages for LLM
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(state.format_for_llm())

        # Get LLM response
        response = self.llm_client.generate(messages)

        # Parse the response to get action
        action = self.parse_action(response)

        # Execute the action
        observation = action.execute(state)

        return action, observation

    def run(self, user_input: str, state: State = None) -> str:
        """Run the agent for a user input.

        Args:
            user_input: The user's message.
            state: Optional existing state, creates new if None.

        Returns:
            str: The final response to the user.
        """
        if state is None:
            state = State()

        # Add user input to state
        state.add_message("user", user_input)

        # Run agent loop
        for i in range(self.max_iterations):
            action, observation = self.step(state)

            # If it's a chat action, we're done
            if isinstance(action, ChatAction):
                return observation

            # Otherwise, continue the loop with the observation
            # The observation is already added to state by action.execute()

        # If we hit max iterations without a chat response, return last observation
        return observation
