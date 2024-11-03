from flask import Flask, request, jsonify, url_for, render_template, session, redirect
import json
from os import environ as env
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
import ERS  # Import the refactored ERS.py for frame analysis
import time
from urllib.parse import urlencode
from collections import Counter
from transformers import pipeline
import random

# Load a Hugging Face model locally

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")

last_request_time = 0  # Epoch time of the last request for rate limiting
emotion_history = []    # List to store detected emotions for averaging

oauth = OAuth(app)
# Auth0 configuration
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

required_env_vars = ["AUTH0_DOMAIN", "AUTH0_CLIENT_ID", "AUTH0_CLIENT_SECRET", "APP_SECRET_KEY"]
for var in required_env_vars:
    if not env.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")


# Route for the main homepage
@app.route("/")
def home():
    if "user" in session:
        # Authenticated users see the main homepage
        return render_template(
            "homepage.html",
            session=session.get("user"),
            pretty=json.dumps(session.get("user"), indent=4),
        )
    else:
        # Unauthenticated users see a public landing page
        return render_template("home.html")

# Callback route for authentication
@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect(url_for("home"))

# Login route
@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(redirect_uri=url_for("callback", _external=True))

# Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN") + "/v2/logout?" +
        urlencode({"returnTo": url_for("home", _external=True), "client_id": env.get("AUTH0_CLIENT_ID")})
    )

# Route for sidebar
@app.route("/sidebar")
def sidebar():
    return render_template("sidebar.html")

generator = pipeline("text-generation", model="gpt2")

fallback_tips = [
    "Take a deep breath and focus on the task at hand.",
    "Remember to take short breaks to stay refreshed.",
    "Stay positive! You've got this.",
    "Break your work into smaller steps and tackle them one at a time.",
    "Make sure to stay hydrated and keep your energy up!"
]

def get_tip(emotion):
    prompt = f"Tips For When Feeling {emotion}:"
    try:
        result = generator(prompt, max_length=50, num_return_sequences=1)
        return result[0]["generated_text"].strip()
    except Exception as e:
        print(f"Error with Hugging Face model: {e}")
        return random.choice(fallback_tips)

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    global last_request_time, emotion_history

    # Check if the frame is provided in the request
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    file = request.files['frame']
    frame_data = file.read()  # Read the frame data

    # Process the frame using ERS to get the dominant emotion
    try:
        emotion_data = ERS.process_frame(frame_data)  # Use the appropriate function from ERS
        dominant_emotion = emotion_data["emotion"]  # Assume ERS returns a dict with the emotion
        print(f"Detected emotion: {dominant_emotion}")  # Debugging line

    except Exception as e:
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

    # Add detected emotion to the history list
    emotion_history.append(dominant_emotion)
    
    # Check if 10 seconds have passed since the last request (for testing purposes)
    current_time = time.time()
    if current_time - last_request_time >= 10:  # Change this back to 300 for production
        # Calculate the most frequent emotion in the history
        if emotion_history:
            most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
            emotion_history.clear()  # Clear history after averaging
            
            # Generate a tip based on the most common emotion
            try:
                tip = get_tip(most_common_emotion)
            except Exception as e:
                tip = f"trouble getting advice...guess you are already amazing ;): {str(e)}"
        else:
            tip = "doing so well, im proud of you <3"

        last_request_time = current_time  # Update last request time
    else:
        tip = "relax and enjoy the day :)"

    return jsonify({'emotion': dominant_emotion, 'tip': tip})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3000))