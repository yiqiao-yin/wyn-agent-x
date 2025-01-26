import json
import os

import wyn_agent_x.list_of_apis  # This will ensure that the API functions are registered
from wyn_agent_x.helper import (
    ChatBot,
    intent_processor,
    load_metadata,
    resolve_and_execute,
)

# Load metadata
metadata_filepath = os.path.join(os.path.dirname(__file__), "metadata.json")
metadata = load_metadata(metadata_filepath)


class AgentX:
    def __init__(
        self,
        openai_api_key: str,
        account_sid: str,
        auth_token: str,
        serpapi_key: str,
        email_key: str,
        claude_api_key: str,
        protocol: str = "You are a helpful agent.",
    ):
        self.event_stream = []
        self.bot = ChatBot(protocol=protocol, api_key=openai_api_key)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.serpapi_key = serpapi_key
        self.email_key = email_key
        self.claude_api_key = claude_api_key

    def start_chat(self):
        # Friendly welcome message with emoji
        print("👋 Welcome! Press 'EXIT' to quit the chat at any time.")

        prompt = input("User: ")

        while "EXIT" not in prompt:
            # Add user message to event_stream
            self.event_stream.append({"event": "user_message", "content": prompt})

            # Process the intent and detect any API calls
            intent_processor(self.event_stream, metadata, self.bot)

            # Get secrets
            secrets = {
                "account_sid": self.account_sid,
                "auth_token": self.auth_token,
                "serpapi_key": self.serpapi_key,
                "email_key": self.email_key,
                "claude_api_key": self.claude_api_key,
            }

            # Check if we need to resolve and execute any API calls
            resolve_and_execute(self.event_stream, metadata, secrets)

            # Get the next user input
            prompt = input("User: ")

        # Exit message with emoji when the user quits
        print("👋 Thanks for chatting! Have a great day! 🌟")
