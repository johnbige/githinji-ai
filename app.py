import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load environment variables from .env (Render supports this natively too)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY")

genai.configure(api_key=api_key)

# Configure model
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="If asked your name, say you are classico A.I developed by John Githinji."
)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())  # Fallback for safety

@app.route("/")
def index():
    session.pop('chat_history', None)
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        history = session.get("chat_history", [])

        chat_session = model.start_chat(
            history=history,
            enable_automatic_function_calling=True
        )

        response = chat_session.send_message(user_message)

        session["chat_history"] = [
            {"role": msg.role, "parts": [part.text for part in msg.parts]}
            for msg in chat_session.history
        ]

        return jsonify({"response": response.text})

    except Exception as e:
        app.logger.exception("Chat processing failed")
        return jsonify({"error": "An internal error occurred."}), 500
