import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# --- Load environment variables and configure the Google Generative AI client ---
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file or environment variables.")

genai.configure(api_key=api_key)

# --- Define Gemini model configuration ---
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="If asked your name, say you are classico A.I developed by John Githinji.",
    tools=[get_date_tool]  # Ensure get_date_tool is defined or imported
)

# --- Flask Web Application Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-mode-secret")

@app.route("/")
def index():
    """Render the main chat page and clear previous session history."""
    session.pop('chat_history', None)
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Process user message and return AI-generated response."""
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        chat_history = session.get('chat_history', [])

        chat_session = model.start_chat(
            history=chat_history,
            enable_automatic_function_calling=True
        )

        response = chat_session.send_message(user_message)

        # Update session history
        session['chat_history'] = [
            {'role': msg.role, 'parts': [part.text for part in msg.parts]}
            for msg in chat_session.history
        ]

        return jsonify({"response": response.text})

    except Exception as e:
        print(f"[ERROR] /chat route exception: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
