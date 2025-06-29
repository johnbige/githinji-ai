import os
import datetime
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# --- Pre-setup: Load environment variables and configure API ---
load_dotenv()

# We can remove the try-except here. If the key is missing, Gunicorn/Flask
# will fail to start and log the error, which is more informative than exit().
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file or environment variables.")
genai.configure(api_key=api_key)

# --- Tool Definition (same as before) ---
def get_todays_date():
    """Returns today's date as a string in YYYY-MM-DD format."""
    return datetime.date.today().isoformat()

# --- Model Configuration (same as before) ---
get_date_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_todays_date",
            description="Gets the current date.",
            parameters={}
        )
    ]
)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="If asked your name, say you are classico A.I developed by John Githinji.",
    tools=[get_date_tool]
)

# --- Flask Web Application ---
app = Flask(__name__)
# IMPORTANT: You need a secret key to use sessions in Flask
# For production, use a long, random, and secret string.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-very-secret-key-for-development")

# We no longer start a global chat session here.
# It will be created on a per-user basis.

@app.route("/")
def index():
    """Renders the main chat page."""
    # Clear session history on new visit for a clean start
    session.pop('chat_history', None)
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message from the user, maintaining user-specific history."""
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        # 1. Retrieve the user's chat history from the session, or start a new one.
        # The history is a list of content objects that the API understands.
        chat_history = session.get('chat_history', [])

        # 2. Create a new chat session *for this request* and load the history.
        chat_session = model.start_chat(
            history=chat_history,
            enable_automatic_function_calling=True
        )

        # 3. Send the new message
        response = chat_session.send_message(user_message)

        # 4. Save the updated history back into the user's session.
        # chat_session.history contains the full conversation (user + model turns).
        session['chat_history'] = [
            {'role': msg.role, 'parts': [part.text for part in msg.parts]}
            for msg in chat_session.history
        ]

        return jsonify({"response": response.text})

    except Exception as e:
        # It's good practice to log the actual error for debugging.
        print(f"An error occurred in /chat: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)