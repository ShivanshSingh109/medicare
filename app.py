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
    get_profile,
    load_profiles
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
HIDE_ADD_EXERCISE_FILE = "hide_add_exercise.json"

def load_user_exercises():
    if not os.path.exists(EXERCISES_FILE):
        return {}
    with open(EXERCISES_FILE, "r") as f:
        return json.load(f)

def save_user_exercises(data):
    with open(EXERCISES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_hide_add_exercise():
    if not os.path.exists(HIDE_ADD_EXERCISE_FILE):
        return {}
    with open(HIDE_ADD_EXERCISE_FILE, "r") as f:
        return json.load(f)

def save_hide_add_exercise(data):
    with open(HIDE_ADD_EXERCISE_FILE, "w") as f:
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
        # Check for duplicate
        if not any(e["name"] == name for e in exercises):
            display_name = name
            if len(name) > 10:
                from webcam import gemini_model
                prompt = f"Give a short, 2-word name for this exercise description: '{name}'. Only return the name."
                try:
                    response = gemini_model.generate_content([prompt])
                    display_name = response.text.strip().split('\n')[0]
                    display_name = display_name.replace('"', '').replace("'", '').strip()
                except Exception as e:
                    display_name = name[:10]
            exercises.append({"name": name, "display_name": display_name})
            data[user_id] = exercises
            save_user_exercises(data)
            return jsonify({"success": True, "display_name": display_name})
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

@app.route("/full_dashboard")
@login_required
def full_dashboard():
    # Clear doctor view if patient is viewing their own dashboard
    session.pop("doctor_view_patient", None)
    return render_template("full_dashboard.html")

@app.route("/exercise_analysis")
@login_required
def exercise_analysis():
    # Clear doctor view if patient is viewing their own analysis
    session.pop("doctor_view_patient", None)
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

    # Doctor view: regenerate analysis in third person
    doctor_patient = session.get("doctor_view_patient")
    if doctor_patient:
        from webcam import _create_analysis_prompt, _attempt_generate, get_user_profile
        # Load pose data for this analysis
        raw_file = None
        data = load_user_exercises()
        exercises = data.get(doctor_patient, [])
        for ex in exercises:
            if "analysis_history" in ex:
                for record in ex["analysis_history"]:
                    if record.get("analysis_file") == analysis_file:
                        raw_file = record.get("raw_file")
                        break
        if raw_file:
            raw_path = os.path.join(JSON_DATA_FOLDER, raw_file)
            if os.path.exists(raw_path):
                with open(raw_path, "r") as f:
                    pose_data = json.load(f)
                patient_profile = get_user_profile(doctor_patient)
                from webcam import _extract_movement_features
                features = _extract_movement_features(pose_data)
                prompt = _create_analysis_prompt(
                    features, analysis_name, patient_profile=patient_profile, third_person=True
                )
                response = _attempt_generate([prompt])
                import json as pyjson
                try:
                    json_start = response.text.find('{')
                    json_end = response.text.rfind('}') + 1
                    analysis_json = pyjson.loads(response.text[json_start:json_end])
                except Exception:
                    analysis_json = {
                        "error": "Model did not return valid JSON.",
                        "raw_response": response.text
                    }
                analysis = analysis_json

    return jsonify({
        "analysis": analysis,
        "exercise_name": analysis_name,
        "performed_at": performed_at
    })

@app.route("/api/full_dashboard_data")
@login_required
def api_full_dashboard_data():
    email = request.args.get("email")
    user_id = email if email else session["user_id"]
    data = load_user_exercises()
    exercises = data.get(user_id, [])
    return jsonify({"exercises": exercises})

@app.route("/api/exercise_progress_report", methods=["POST"])
@login_required
def exercise_progress_report():
    req = request.get_json()
    exercise_name = req.get("name", "")

    doctor_patient = session.get("doctor_view_patient")
    if doctor_patient:
        user_id = doctor_patient
        third_person = True
    else:
        user_id = session["user_id"]
        third_person = False

    data = load_user_exercises()
    exercises = data.get(user_id, [])
    analysis_files = []
    for ex in exercises:
        if ex["name"] == exercise_name and "analysis_history" in ex:
            for record in ex["analysis_history"]:
                analysis_files.append(record["analysis_file"])
    if not analysis_files:
        return jsonify({"error": "No analysis data found for this exercise."}), 404

    # Progress report cache file - include prompt style
    cache_dir = "progress_reports"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"{user_id}_{exercise_name.replace(' ', '_')}_{'doctor' if third_person else 'patient'}.json"
    )

    # Check cache
    cached = None
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
        cached_files = set(cached.get("analysis_files", []))
        if cached_files == set(analysis_files):
            return jsonify(cached["report"])

    # Load all analysis JSONs
    analysis_data = []
    for fname in analysis_files:
        fpath = os.path.join(JSON_DATA_FOLDER, fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                analysis_data.append(json.load(f))

    # Build prompt for LLM
    if third_person:
        prompt = f"""You are an expert physiotherapy coach. Here are multiple AI analyses for the exercise '{exercise_name}' performed by a patient. Please summarize the patient's progress over time, highlight improvements, recurring issues, and give encouragement. Use third person ("they", "the patient") in your feedback. Always refer to the patient in third person (never use 'you' or 'your'). Be concise and clear.

Analysis history:
{json.dumps(analysis_data, indent=2)}

Additionally, analyze the feedback and provide pie chart data as JSON showing the percentage distribution of feedback categories (e.g. 'good form', 'needs improvement', 'speed issues', 'consistency', etc). 
Return your answer as:
{{
  "progress_report": "<short summary>",
  "pie_chart_data": {{
    "Good Form": <number>,
    "Needs Improvement": <number>,
    "Speed Issues": <number>,
    "Consistency": <number>,
    "Other": <number>
  }}
}}
"""
    else:
        prompt = f"""You are an expert physiotherapy coach. Here are multiple AI analyses for the exercise '{exercise_name}'. Please summarize the user's progress over time, highlight improvements, recurring issues, and give encouragement. Be concise and clear.

Analysis history:
{json.dumps(analysis_data, indent=2)}

Additionally, analyze the feedback and provide pie chart data as JSON showing the percentage distribution of feedback categories (e.g. 'good form', 'needs improvement', 'speed issues', 'consistency', etc). 
Return your answer as:
{{
  "progress_report": "<short summary>",
  "pie_chart_data": {{
    "Good Form": <number>,
    "Needs Improvement": <number>,
    "Speed Issues": <number>,
    "Consistency": <number>,
    "Other": <number>
  }}
}}
"""

    from webcam import gemini_model
    try:
        response = gemini_model.generate_content([prompt])
        import re
        match = re.search(r"\{[\s\S]*\}", response.text)
        if match:
            result_json = json.loads(match.group(0))
        else:
            result_json = {"progress_report": response.text, "pie_chart_data": {}}
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump({"analysis_files": analysis_files, "report": result_json}, f, indent=2)
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

@app.route("/api/hide_add_exercise", methods=["POST"])
@login_required
def hide_add_exercise():
    user_id = session["user_id"]  # <-- Correct key!
    data = load_hide_add_exercise()
    data[user_id] = True
    save_hide_add_exercise(data)
    return jsonify({"success": True})

@app.route("/api/hide_add_exercise", methods=["GET"])
@login_required
def get_hide_add_exercise():
    user_id = session["user_id"]
    data = load_hide_add_exercise()
    return jsonify({"hide": data.get(user_id, False)})

@app.route("/api/raw_pose_data")
@login_required
def api_raw_pose_data():
    analysis_file = session.get("selected_analysis_file", "")
    user_id = session["user_id"]
    # Find the raw_file from user_exercises.json
    data = load_user_exercises()
    exercises = data.get(user_id, [])
    raw_file = None
    for ex in exercises:
        if "analysis_history" in ex:
            for record in ex["analysis_history"]:
                if record.get("analysis_file") == analysis_file:
                    raw_file = record.get("raw_file")
                    break
    if not raw_file:
        return jsonify({"error": "Raw pose data not found"}), 404
    raw_path = os.path.join(JSON_DATA_FOLDER, raw_file)
    if not os.path.exists(raw_path):
        return jsonify({"error": "Raw file not found"}), 404
    with open(raw_path, "r") as f:
        pose_data = json.load(f)
    return jsonify({"pose_data": pose_data})

@app.route("/doctor")
def doctor_dashboard():
    return render_template("doctor_dashboard.html")

@app.route("/api/all_patients")
def api_all_patients():
    profiles = load_profiles()
    return jsonify({"patients": list(profiles.values())})

@app.route("/doctor/patient/<email>")
def doctor_patient_dashboard(email):
    # Set session for doctor view
    session["doctor_view_patient"] = email
    return render_template("doctor_patient_dashboard.html")

@app.route("/doctor/analysis/<email>/<analysis_file>")
def doctor_analysis(email, analysis_file):
    # Set doctor view session for this patient
    session["doctor_view_patient"] = email
    session["selected_analysis_file"] = analysis_file
    # Optionally, you can also set selected_analysis_name and performed_at if you want
    return render_template("doctor_analysis.html")

@app.route("/api/doctor_view_patient")
def api_doctor_view_patient():
    email = session.get("doctor_view_patient")
    return jsonify({"email": email})

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
