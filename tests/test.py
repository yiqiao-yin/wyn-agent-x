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
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


# Print loaded keys (for demonstration purpose, avoid printing secrets in production)
# print("OpenAI API Key:", openai_api_key)
# print("Twilio Account SID:", twilio_account_sid)
# print("Twilio Auth Token:", twilio_auth_token)
# print("SerpAPI API Key:", serpapi_api_key)
# print("PY Email Key:", py_email_key)


# Import the agent
from wyn_agent_x.main import AgentX

# Initialize and start the chat!
agent = AgentX(
    openai_api_key=OPENAI_API_KEY,
    account_sid=TWILIO_ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    serpapi_key=SERPAPI_API_KEY,
    email_key=PY_EMAIL_KEY,
    claude_api_key=CLAUDE_API_KEY,
    protocol="You are a helpful assistant.",
)

# Run the chat
agent.start_chat()
