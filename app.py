import os
import json
import uuid
import tempfile
import threading
import time
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    send_from_directory,
)
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- App & Global State ---
app = Flask(
    __name__,
    template_folder="data/templates",
    static_folder="data/static",
)

# --- Constants ---
JSON_DATA_FOLDER = "analysis_data"

# --- Gemini Configuration ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure it's set in your .env file.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/models/<path:filename>")
def serve_model(filename):
    """Serves the MediaPipe model file to the client."""
    return send_from_directory(".", filename)

@app.route("/analyze_exercise", methods=["POST"])
def analyze_exercise():
    """Receives pose data, saves it, and uses Gemini for analysis."""
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Invalid request."}), 400

    pose_data = request_data.get("pose_data")
    user_description = request_data.get("description", "an exercise").strip()
    form_prompt_template = request_data.get("form_prompt")

    if not pose_data:
        return jsonify({"error": "No pose_data provided."}), 400
    if not user_description:
        user_description = "an exercise"

    # Ensure form_prompt_template has a default if missing
    if not form_prompt_template:
        form_prompt_template = "Analyze the exercise '{user_description}' based on this data:\n{truncated_json_data}"

    try:
        # --- Save the incoming pose data to a file for debugging ---
        filename = f"{uuid.uuid4()}.json"
        filepath = os.path.join(JSON_DATA_FOLDER, filename)
        with open(filepath, 'w') as f:
            json.dump(pose_data, f, indent=2)
        print(f"[app] Saved incoming pose data to '{filepath}'")

        # --- Use Gemini File API to send large data ---
        # Instead of pasting JSON string into prompt, we upload it as a file.
        
        # 1. Create a temporary file for upload
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            json.dump(pose_data, tmp_file)
            tmp_file_path = tmp_file.name

        print(f"[app] Uploading {len(pose_data)} frames to Gemini via File API...")
        
        # 2. Upload the file
        # Changed mime_type to 'text/plain' because 'application/json' is not supported for generation
        uploaded_file = genai.upload_file(tmp_file_path, mime_type="text/plain")
        
        # 3. Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            print("[app] Waiting for file processing...")
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise ValueError("File upload failed.")

        print(f"[app] File uploaded: {uploaded_file.uri}")

        # 4. Prepare Prompt
        # Replace description
        prompt_text = form_prompt_template.replace('{user_description}', user_description)
        # Remove the JSON placeholder since we are attaching the file
        prompt_text = prompt_text.replace('{truncated_json_data}', "(See attached JSON file)")

        print("[gemini] Sending prompt + file to Gemini model...")
        
        # 5. Generate Content with File Attachment
        response = gemini_model.generate_content([prompt_text, uploaded_file])
        
        # 6. Cleanup local temp file
        os.unlink(tmp_file_path)

        analysis_text = response.text.strip()

        analysis_json = {}
        try:
            # Attempt to find and parse the JSON object from the response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response.")

            clean_json_str = analysis_text[json_start:json_end]
            analysis_json = json.loads(clean_json_str)

            # Ensure rep_count is an integer, default to 0 if missing or invalid
            analysis_json['repetition_count'] = int(analysis_json.get('repetition_count', 0))

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # If parsing fails, create a fallback object with the raw text
            print(f"[gemini] Warning: Could not parse analysis JSON. Error: {e}")
            analysis_json = {
                "error": "The model did not return a valid JSON object.",
                "raw_response": analysis_text,
                "repetition_count": 0  # Provide a default for the UI
            }

        print(f"[gemini] Combined analysis result: {analysis_json}")
        return jsonify(analysis_json)

    except Exception as e:
        print(f"[gemini] Error during analysis: {e}")
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500

if __name__ == "__main__":
    # --- Create folder for saving JSON data ---
    os.makedirs(JSON_DATA_FOLDER, exist_ok=True)
    print(f"[app] Data will be saved in '{JSON_DATA_FOLDER}' directory.")

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
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            exit(1)

    print("[app] Flask server starting.")
    app.run(host="0.0.0.0", port=5000)
