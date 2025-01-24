import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PY_EMAIL_KEY = os.getenv("PY_EMAIL_KEY")


# Import the agent
from wyn_agent_x.main import AgentX

# Initialize and start the chat!
agent = AgentX(
    api_key=OPENAI_API_KEY,
    account_sid=TWILIO_ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    serpapi_key=SERPAPI_API_KEY,
    email_key=PY_EMAIL_KEY,
    protocol="You are a helpful assistant.",
)

# Run the chat
agent.start_chat()
