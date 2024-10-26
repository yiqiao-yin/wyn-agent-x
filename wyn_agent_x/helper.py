import re
from typing import Any, Dict, Optional, Callable, List
import json
from openai import OpenAI

# Load metadata from JSON file
def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load metadata from a given JSON file path.
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return metadata

class ChatBot:
    def __init__(self, protocol: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.protocol = protocol
        self.history = [{"role": "system", "content": self.protocol}]

    def generate_response(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.history
        )

        response = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_history(self) -> list:
        return self.history

def match_trigger_words(user_message: str, trigger_words: List[str]) -> bool:
    """
    Check if any of the trigger words are present in the user's message.
    Returns True if a match is found, otherwise False.
    """
    for trigger in trigger_words:
        if re.search(rf"\b{trigger.lower()}\b", user_message.lower()):
            return True
    return False

def intent_processor(event_stream: List[Dict[str, Any]], metadata: Dict[str, Any], bot: ChatBot) -> None:
    """
    Process the event stream, detect any intent using metadata trigger words,
    and append an API call to the event stream if a trigger word is detected.
    """
    # Find the latest user message from the event stream
    last_event = next((event for event in reversed(event_stream) if event.get("event") == "user_message"), None)

    if not last_event:
        print("No user message found in event stream.")
        return

    user_message = last_event.get("content", "")

    # Iterate through all functions in metadata and check if trigger words are present
    for api_name, api_metadata in metadata.items():
        trigger_words = api_metadata.get("trigger_word", [])

        # Check if any trigger word matches the user message
        if match_trigger_words(user_message, trigger_words):
            print(f"Intent detected for API call: {api_name}")

            # Append the API call to the event stream
            event_stream.append({
                "intent_processor": "api_call",
                "api_name": api_name,
                "response": {"status": "none"}
            })
            break
    else:
        # No intent detected, use the bot for a general conversation
        bot_response = bot.generate_response(user_message)
        print(f"Bot Response: {bot_response}")
        event_stream.append({"event": "assistant_message", "content": bot_response})

# Function registry to map API names to actual function calls
function_registry: Dict[str, Callable[..., Dict[str, Any]]] = {}

def register_function(api_name: str):
    """
    Decorator to register API functions dynamically into the function registry.
    """
    def decorator(func: Callable[..., Dict[str, Any]]):
        function_registry[api_name] = func
        return func
    return decorator

def find_in_event_stream(key: str, event_stream: List[Dict[str, Any]], current_api_name: str) -> Optional[str]:
    """
    Helper function to find a piece of information in the event stream.
    Searches in reverse order for the most recent occurrence of a matching key.
    Stops when any 'success' status is encountered, meaning that information is already used.
    
    :param key: The key to search for.
    :param event_stream: The event stream containing user and API call events.
    :param current_api_name: The name of the current API call to handle context-specific data.
    :return: The content of the user message matching the key.
    """
    for event in reversed(event_stream):
        # Stop the search if an API call with a 'success' status is found for any API
        if event.get("event") == "api_call" and "success" in event.get("response", {}).get("status", ""):
            break
        
        # Continue searching user messages
        if event.get("event") == "user_message" and key.lower() in event.get("content", "").lower():
            return event.get("content")
    
    return None

def resolve_and_execute(event_stream: List[Dict[str, Any]], metadata: Dict[str, Any], secrets: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Resolve dependencies and execute the specified API call dynamically.
    Only runs if the last event in the event_stream has 'intent_processor': 'api_call'.
    """
    api_event = next((event for event in reversed(event_stream) if event.get("intent_processor") == "api_call"), None)

    if not api_event:
        print("No API call found in the event stream. Exiting.")
        return None

    api_name = api_event.get("api_name")
    if not api_name or api_name not in metadata:
        print(f"Error: API '{api_name}' is not defined in the metadata.")
        return None

    payload = {}
    for key in metadata[api_name]["sample_payload"]:
        # Pass the current API name to maintain context
        value = find_in_event_stream(key, event_stream, current_api_name=api_name)
        if not value:
            value = input(f"Please provide {key}: ")
            event_stream.append({"event": "user_message", "content": f"My {key} is {value}."})
        payload[key] = value.split(" is ")[-1]

    api_function = function_registry.get(api_name)
    if api_function:
        return api_function(payload, secrets, event_stream)  # Pass event_stream as argument
    else:
        print(f"Error: No function found for API call '{api_name}'.")
        return None
