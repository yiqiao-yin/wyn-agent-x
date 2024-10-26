from typing import Any, Dict, List

from serpapi import GoogleSearch
from twilio.rest import Client

from wyn_agent_x.helper import register_function


@register_function("send_sms")
def send_sms(
    payload: Dict[str, str], secrets: Dict[str, str], event_stream: list
) -> Dict[str, Any]:
    """
    Simulate sending an SMS using the Twilio API.

    Args:
        payload (Dict[str, str]): Contains the message details and recipient information.
        secrets (Dict[str, str]): Contains sensitive data like account SID and auth token.
        event_stream (list): A list to log events and responses for the SMS API call.

    Returns:
        Dict[str, Any]: A dictionary containing the status and result of the SMS operation.

    The function limits the message body to 1600 characters.
    If exceeded, it returns an error response without sending the SMS.
    """
    print(f"👀 API Call: Sending SMS with payload: {payload}")

    # Extract Secrets
    account_sid = secrets["account_sid"]
    auth_token = secrets["auth_token"]

    # Message body check for character limit
    message_body = (
        f"Hello {payload['name']}, here's the message: {payload['message body']}"
    )
    if len(message_body) > 1600:
        response = {
            "status": "error",
            "message": "Message body exceeds 1600 character limit.",
            "model_name": "None",
        }
        print("👀 Error: Message body exceeds the 1600 character limit.")

        # Append the error result to the event stream
        event_stream.append(
            {"event": "api_call", "api_name": "send_sms", "response": response}
        )
        return response

    try:
        # Initialize Twilio Client with provided credentials
        client = Client(account_sid, auth_token)

        # Simulate sending a message (replace with actual logic for real SMS sending)
        message = client.messages.create(
            body=message_body,
            from_="+18552060350",  # Replace with a valid Twilio number
            to="+15859538396",  # Replace with the destination number
        )

        print(f"👀 Message SID: {message.sid}")
        response = {"status": f"success: {message.sid}", "model_name": "None"}

    except Exception as e:
        # Gracefully handle any exceptions
        response = {
            "status": f"error: {str(e)}",
            "message": "Failed to send SMS due to an error.",
            "model_name": "None",
        }
        print(f"👀 Error: {str(e)}")

    # Append the result to the event stream
    event_stream.append(
        {"event": "api_call", "api_name": "send_sms", "response": response}
    )

    return response


# Register the google_search function
@register_function("google_search")
def google_search(
    payload: Dict[str, str], secrets: Dict[str, str], event_stream: list
) -> Dict[str, Any]:
    """
    Simulate a Google search using the SerpAPI and return the results formatted as a Markdown table.

    Args:
        payload (Dict[str, str]): A dictionary containing the search parameters:
                                  - query: The search query string.
                                  - location: The location for the search.
        secrets (Dict[str, str]): Contains the SerpAPI key for API authentication.
        event_stream (list): A list used to log events and responses.

    Returns:
        Dict[str, Any]: A dictionary with the status of the API call and a Markdown table of search results.
    """

    # Extract Secrets
    serpapi_key = secrets["serpapi_key"]

    # Extract search parameters from the payload
    query = payload.get("query", "")
    location = payload.get("location", "")

    # Checkpoint
    print(f"Run google search API using: \nquery={query}, \nlocation={location}")

    try:
        # Perform the Google Search using SerpAPI
        search = GoogleSearch(
            {"q": query, "location": location, "api_key": serpapi_key}
        )
        results = search.get_dict()

        # Extract relevant results
        organic_results = results.get("organic_results", [])

        # Convert search results to a Markdown table
        md_table = "| Title | Link | Snippet |\n| :--- | :--- | :--- |\n"
        for item in organic_results:
            title = item.get("title", "N/A")
            link = item.get("link", "#")
            snippet = item.get("snippet", "N/A")
            md_table += f"| {title} | [Link]({link}) | {snippet} |\n"
        print(f"👀 Results: \n{md_table}")

        # Prepare the response
        response = {"status": "success", "model_name": "None", "results": md_table}

        # Append the result to the event stream
        event_stream.append(
            {"event": "api_call", "api_name": "google_search", "response": response}
        )

    except Exception as e:
        # Handle any exceptions that occur during the API call
        response = {"status": f"error: {str(e)}", "model_name": "None"}
        event_stream.append(
            {"event": "api_call", "api_name": "google_search", "response": response}
        )

    return response
