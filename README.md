# 🌟 WYN-Agent-X

WYN-Agent-X is a dynamic and extendable chatbot that integrates with OpenAI and Twilio, allowing you to seamlessly handle user intents and trigger APIs (like sending SMS) based on natural language input. Plus, it’s super friendly and conversational! 🤖💬

### Features:
- **AI-Powered Conversations**: Uses OpenAI’s GPT models for general chit-chat. Just type away!
- **Trigger-Based API Calls**: Automatically detects when users want to perform specific tasks (like sending a message), and triggers the corresponding API call.
- **Easy to Extend**: Add more APIs by simply updating the `metadata.json` file. No need to dig into the core logic! 🌱

---

## 🚀 Installation

You can easily install the package via `pip`:

```bash
pip install wyn-agent-x
```

---

## 📂 Directory Structure

```bash
wyn-agent-x/
│
├── requirements.txt     # List of dependencies to install
├── wyn_agent_x/
│   ├── __init__.py      # Initializes the package
│   ├── main.py          # Main entry point for the chatbot
│   ├── helper.py        # Helper functions and processing logic
│   ├── list_of_apis.py  # All API functions registered here
│   ├── metadata.json    # Dynamic metadata for API calls and trigger words
│── pyproject.toml       # Optional config if packaging the project
└── README.md            # You're reading this!
```

---

## 🎮 Sample Usage

Want to try it out? Just import the `AgentX` class, provide your API keys, and start chatting with your agent!

```python
from google.colab import userdata

# Fetch API credentials
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = userdata.get("YOUR_TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = userdata.get("YOUR_TWILIO_AUTH_TOKEN")

# Import the agent
from wyn_agent_x.main import AgentX

# Initialize and start the chat!
agent = AgentX(api_key=OPENAI_API_KEY, account_sid=TWILIO_ACCOUNT_SID, auth_token=TWILIO_AUTH_TOKEN)
agent.start_chat()
```

Once started, you'll see this friendly message:
```
👋 Welcome! Press 'EXIT' to quit the chat at any time.
```

Feel free to chat with the bot, ask it to send messages, or perform any task you've configured in the metadata. When you're done, simply type `EXIT` to end the session with a friendly goodbye! 👋

---

## 📖 How it Works

1. **Intent Detection**: The agent listens for specific trigger words (like "send a message" or "set a demo") from user input and matches them against the triggers defined in `metadata.json`.
   
2. **API Calls**: When an intent is detected (e.g., sending an SMS), it calls the corresponding API (like Twilio's SMS API) and logs the event in the `event_stream`.

3. **Dynamic Functions**: Adding a new API or intent is as simple as updating the `metadata.json` file and registering the new API in `list_of_apis.py`. No need to modify core logic! 🚀

---

## 🛠️ Extend and Customize

You can easily extend WYN-Agent-X by adding new API calls or intents:

1. **Update `metadata.json`** with new API information and trigger words:
   ```json
   {
       "send_email": {
           "trigger_word": ["send email", "notify via email"],
           "sample_payload": {"email": "string", "subject": "string"},
           "prerequisite": null
       }
   }
   ```

2. **Register your new API** in `list_of_apis.py` with a simple decorator:
   ```python
   @register_function("send_sms")
    def send_sms(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
       # Code to send email goes here!
       pass
   ```

---

## 📜 License

MIT License - Enjoy, use, and extend this project freely! 🥳

---

## 👤 Author

**Yiqiao Yin**

📧 Email: eagle0504@gmail.com

Feel free to reach out if you have any questions, suggestions, or just want to say hi! 😊
