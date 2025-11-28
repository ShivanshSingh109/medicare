import os
import json
from flask import Flask
from dotenv import load_dotenv
from flask import session, render_template, jsonify, request

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
    profile_setup_page,
    register,
    login,
    logout,
    get_current_user,
    login_required,
    profile_required,
    save_profile,
    update_profile,
    get_profile
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
app.add_url_rule("/profile-setup", "profile_setup_page", profile_setup_page)
app.add_url_rule("/api/register", "register", register, methods=["POST"])
app.add_url_rule("/api/login", "login", login, methods=["POST"])
app.add_url_rule("/api/profile", "save_profile", save_profile, methods=["POST"])
app.add_url_rule("/api/profile", "get_profile", get_profile, methods=["GET"])
app.add_url_rule("/api/profile", "update_profile", update_profile, methods=["PUT"])
app.add_url_rule("/logout", "logout", logout)
app.add_url_rule("/api/user", "get_current_user", get_current_user)

# Register main routes (protected - requires both login AND profile)
app.add_url_rule("/", "index", profile_required(index))
app.add_url_rule("/models/<path:filename>", "serve_model", serve_model)
app.add_url_rule("/analyze_exercise", "analyze_exercise", profile_required(analyze_exercise), methods=["POST"])

EXERCISES_FILE = "user_exercises.json"

def load_user_exercises():
    if not os.path.exists(EXERCISES_FILE):
        return {}
    with open(EXERCISES_FILE, "r") as f:
        return json.load(f)

def save_user_exercises(data):
    with open(EXERCISES_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/exercise_dashboard")
@login_required
def exercise_dashboard():
    return render_template("exercise_dashboard.html")

@app.route("/api/exercises", methods=["GET", "POST"])
@login_required
def api_exercises():
    user_id = session["user_id"]
    data = load_user_exercises()
    if request.method == "GET":
        return jsonify({"exercises": data.get(user_id, [])})
    else:
        req = request.get_json()
        name = req.get("name", "").strip()
        if not name:
            return jsonify({"error": "No exercise name"}), 400
        exercises = data.get(user_id, [])
        if not any(e["name"] == name for e in exercises):
            exercises.append({"name": name})
            data[user_id] = exercises
            save_user_exercises(data)
        return jsonify({"success": True})

@app.route("/api/select_exercise", methods=["POST"])
@login_required
def api_select_exercise():
    req = request.get_json()
    session["selected_exercise"] = req.get("name", "")
    return jsonify({"success": True})

@app.route("/api/selected_exercise")
@login_required
def api_selected_exercise():
    name = session.get("selected_exercise", "")
    username = session.get("username", "")
    return jsonify({"name": name, "username": username})

@app.route("/exercise_tracker")
@login_required
def exercise_tracker():
    return render_template("exercise_tracker.html")

@app.route("/api/save_exercise_analysis", methods=["POST"])
@login_required
def save_exercise_analysis():
    user_id = session["user_id"]
    req = request.get_json()
    exercise_name = req.get("name", "")
    analysis_file = req.get("analysis_file", "")

    data = load_user_exercises()
    exercises = data.get(user_id, [])
    for ex in exercises:
        if ex["name"] == exercise_name:
            # Create history list if not present
            if "analysis_history" not in ex:
                ex["analysis_history"] = []
            # Add new record
            ex["analysis_history"].append({
                "performed_at": __import__('datetime').datetime.now().isoformat(),
                "analysis_file": analysis_file
            })
            # Remove quick access fields if present
            ex.pop("last_performed", None)
            ex.pop("analysis_file", None)
    data[user_id] = exercises
    save_user_exercises(data)
    return jsonify({"success": True})

import threading

def background_analysis(user_id, exercise_name, pose_data):
    from webcam import run_analysis_logic, get_user_profile
    import uuid

    with app.app_context():
        patient_profile = get_user_profile(user_id)

        # Save raw pose data to a unique file
        raw_filename = f"raw_{uuid.uuid4().hex}.json"
        raw_filepath = os.path.join(JSON_DATA_FOLDER, raw_filename)
        with open(raw_filepath, "w") as f:
            json.dump(pose_data, f, indent=2)

        # Run analysis and get analysis file name
        analysis_json, analysis_file = run_analysis_logic(
            pose_data, exercise_name, patient_profile
        )

        # Save both filenames in history
        data = load_user_exercises()
        exercises = data.get(user_id, [])
        for ex in exercises:
            if ex["name"] == exercise_name:
                if "analysis_history" not in ex:
                    ex["analysis_history"] = []
                ex["analysis_history"].append({
                    "performed_at": __import__('datetime').datetime.now().isoformat(),
                    "analysis_file": analysis_file,
                    "raw_file": raw_filename
                })
        data[user_id] = exercises
        save_user_exercises(data)

@app.route("/api/queue_exercise_analysis", methods=["POST"])
@login_required
def queue_exercise_analysis():
    user_id = session["user_id"]
    req = request.get_json()
    exercise_name = req.get("name", "")
    pose_data = req.get("pose_data", [])
    # Start background thread for analysis
    threading.Thread(target=background_analysis, args=(user_id, exercise_name, pose_data)).start()
    return jsonify({"success": True})

@app.route("/api/select_analysis", methods=["POST"])
@login_required
def api_select_analysis():
    req = request.get_json()
    session["selected_analysis_file"] = req.get("analysis_file", "")
    session["selected_analysis_name"] = req.get("name", "")
    session["selected_analysis_date"] = req.get("performed_at", "")
    return jsonify({"success": True})

@app.route("/exercise_analysis")
@login_required
def exercise_analysis():
    return render_template("exercise_analysis.html")

@app.route("/api/analysis_data")
@login_required
def api_analysis_data():
    analysis_file = session.get("selected_analysis_file", "")
    analysis_name = session.get("selected_analysis_name", "")
    performed_at = session.get("selected_analysis_date", "")
    if not analysis_file:
        return jsonify({"error": "No analysis selected"}), 400
    analysis_path = os.path.join(JSON_DATA_FOLDER, analysis_file)
    if not os.path.exists(analysis_path):
        return jsonify({"error": "Analysis file not found"}), 404
    with open(analysis_path, "r") as f:
        analysis = json.load(f)
    return jsonify({
        "analysis": analysis,
        "exercise_name": analysis_name,
        "performed_at": performed_at
    })

@app.route("/full_dashboard")
@login_required
def full_dashboard():
    return render_template("full_dashboard.html")

@app.route("/api/full_dashboard_data")
@login_required
def api_full_dashboard_data():
    user_id = session["user_id"]
    data = load_user_exercises()
    exercises = data.get(user_id, [])
    return jsonify({"exercises": exercises})

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
