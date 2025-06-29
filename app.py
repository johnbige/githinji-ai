import os
import datetime
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- Pre-setup: Load environment variables and configure API ---
# Load the API key from the .env file
load_dotenv()

try:
    # Configure the Gemini API with the key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=api_key)

except ValueError as e:
    print(e)
    # Exit if the API key is not configured
    exit()

# --- Tool Definition: A function the model can call ---
def get_todays_date():
    """Returns today's date as a string in YYYY-MM-DD format."""
    return datetime.date.today().isoformat()

# --- Model Configuration (do this once at startup) ---
# Define the tool for the model
get_date_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_todays_date",
            description="Gets the current date.",
            parameters={} # No parameters needed
        )
    ]
)

# Create the Generative Model instance
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    system_instruction="If asked your name, say you are classico A.I developed by John Githinji.",
    tools=[get_date_tool]
)

# --- Flask Web Application ---
app = Flask(__name__)

# This is a simple in-memory store for chat history.
# For a real-world app, you'd use a database or user sessions.
chat_session = model.start_chat(
    enable_automatic_function_calling=True
)

@app.route("/")
def index():
    """Renders the main chat page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message from the user."""
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        # Send the message to the model and get the response
        response = chat_session.send_message(user_message)

        # Return the model's text response as JSON
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)