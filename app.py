import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

from webcam import (
    index,
    serve_model,
    analyze_exercise,
    init_gemini,
    JSON_DATA_FOLDER,
)

from auth import (
    login_page,
    register_page,
    register,
    login,
    logout,
    get_current_user,
    login_required
)

app = Flask(
    __name__,
    template_folder="data/templates",
    static_folder="data/static",
)

# Secret key for session management
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")

# Initialize Gemini model
init_gemini()

# Register authentication routes
app.add_url_rule("/login", "login_page", login_page)
app.add_url_rule("/register", "register_page", register_page)
app.add_url_rule("/api/register", "register", register, methods=["POST"])
app.add_url_rule("/api/login", "login", login, methods=["POST"])
app.add_url_rule("/logout", "logout", logout)
app.add_url_rule("/api/user", "get_current_user", get_current_user)

# Register main routes (protected)
app.add_url_rule("/", "index", login_required(index))
app.add_url_rule("/models/<path:filename>", "serve_model", serve_model)
app.add_url_rule("/analyze_exercise", "analyze_exercise", login_required(analyze_exercise), methods=["POST"])

if __name__ == "__main__":
    os.makedirs(JSON_DATA_FOLDER, exist_ok=True)
    print(f"[app] Data directory: '{JSON_DATA_FOLDER}'")

    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        try:
            import requests
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(r.content)
            print("Model downloaded.")
        except Exception as e:
            print(f"Download error: {e}")
            exit(1)

    print("[app] Flask server starting.")
    app.run(host="0.0.0.0", port=5000)
