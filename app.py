import os
import cv2
import time
import threading
from datetime import datetime
import numpy as np
import io
import json
from queue import Queue, Empty
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
import tempfile
from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    send_file,
    request,
)

import mediapipe as mp
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

# Load environment variables from .env file
load_dotenv()

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- Configuration ---
LANDMARKS_TO_RECORD = {
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}
VISIBILITY_THRESHOLD = 0.5

# --- App & Global State ---
app = Flask(
    __name__,
    template_folder="data/templates",
    static_folder="data/static",
)

# Create a temporary directory for session data
TEMP_DIR = tempfile.mkdtemp(prefix="pose_data_")
print(f"[app] Storing temporary session files in: {TEMP_DIR}")

class AppState:
    def __init__(self):
        self.tracking_enabled = False
        self.latest_processed_jpeg = None
        self.lock = threading.Lock()
        self.current_session_uid = None
        self.current_session_file_path = None

STATE = AppState()
frame_queue = Queue(maxsize=2) # Increased queue size for smoother streaming

# --- Gemini Configuration ---
# Load API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure it's set in your .env file.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

# --- MediaPipe & Camera Management ---
def get_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
        running_mode=VisionRunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    print("[mediapipe] PoseLandmarker initialized.", flush=True)
    return landmarker

# --- Background Workers ---
def pose_processor_worker():
    print("[pose_processor] Starting pose processor.", flush=True)
    landmarker = get_landmarker()
    
    while True:
        try:
            # Get frame from the queue populated by the /upload_frame endpoint
            frame = frame_queue.get(timeout=2)
        except Empty:
            # If the queue is empty, create a placeholder image
            frame = np.zeros((480, 640, 3), dtype="uint8")
            cv2.putText(frame, "Waiting for client frames...", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # The frame from JS is already flipped, so no need for cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = landmarker.detect(mp_image)

        if results.pose_landmarks:
            for pose_landmarks_list in results.pose_landmarks:
                pose_landmark_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmark_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks_list
                ])
                mp_drawing.draw_landmarks(
                    frame, pose_landmark_proto, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                with STATE.lock:
                    if STATE.tracking_enabled and STATE.current_session_file_path:
                        # Record data for every processed frame without a delay
                        ts = datetime.now().isoformat()
                        row = {"timestamp": ts}
                        for i, lm in enumerate(pose_landmarks_list):
                            if i in LANDMARKS_TO_RECORD:
                                landmark_name = mp_pose.PoseLandmark(i).name
                                if lm.visibility < VISIBILITY_THRESHOLD:
                                    row[landmark_name] = None
                                else:
                                    row[landmark_name] = f"{lm.x:.4f},{lm.y:.4f},{lm.z:.4f}"
                        
                        # Append as a JSON line to the session file
                        try:
                            with open(STATE.current_session_file_path, "a") as f:
                                f.write(json.dumps(row) + "\n")
                        except Exception as e:
                            print(f"[pose_processor] Error writing to file: {e}")

        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            with STATE.lock:
                STATE.latest_processed_jpeg = buffer.tobytes()

# --- Frame Generator for Streaming ---
def gen_frames():
    while True:
        with STATE.lock:
            frame_bytes = STATE.latest_processed_jpeg
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """Receives a frame from the browser, decodes it, and puts it in the queue."""
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame uploaded"}), 400
    
    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400

        # Put the decoded frame into the queue for the processor worker
        if frame_queue.full():
            frame_queue.get_nowait()  # Discard oldest frame if queue is full
        frame_queue.put(frame)

        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[upload_frame] Error processing frame: {e}")
        return jsonify({"error": "Failed to process frame"}), 500

@app.route("/start_tracking", methods=["POST"])
def start_tracking():
    with STATE.lock:
        session_uid = str(uuid.uuid4())
        session_file_path = os.path.join(TEMP_DIR, f"{session_uid}.jsonl")

        STATE.current_session_uid = session_uid
        STATE.current_session_file_path = session_file_path
        STATE.tracking_enabled = True
        
        # Create the file
        with open(session_file_path, "w") as f:
            pass # Create an empty file

    status_msg = "Tracking started. Recording data continuously."
    print(f"[tracking] {status_msg} Session: {session_uid}", flush=True)
    return jsonify({"status": status_msg, "session_uid": session_uid})

@app.route("/stop_tracking", methods=["POST"])
def stop_tracking():
    session_uid = None
    with STATE.lock:
        STATE.tracking_enabled = False
        session_uid = STATE.current_session_uid
        STATE.current_session_uid = None
        STATE.current_session_file_path = None

    if session_uid:
        print(f"[tracking] Stopped session: {session_uid}", flush=True)
        return jsonify({"status": f"Tracking stopped for session {session_uid}."})
    else:
        return jsonify({"status": "Tracking was not active."})

@app.route("/analyze_exercise", methods=["POST"])
def analyze_exercise():
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Invalid request."}), 400

    session_uid = request_data.get("session_uid")
    if not session_uid:
        return jsonify({"error": "No session_uid provided."}), 400

    session_file_path = os.path.join(TEMP_DIR, f"{session_uid}.jsonl")

    if not os.path.exists(session_file_path):
        return jsonify({"error": "Session data not found. It may have been analyzed already."}), 404

    pose_rows = []
    try:
        with open(session_file_path, "r") as f:
            for line in f:
                pose_rows.append(json.loads(line))
        
        if not pose_rows:
            return jsonify({"error": "No data to analyze."}), 400

        # Get the user's exercise description from the request
        user_description = request_data.get("description", "an exercise").strip()
        if not user_description:
            user_description = "an exercise"  # Fallback if description is empty

        # Prepare and truncate the data once for both prompts
        json_data = json.dumps(pose_rows, indent=2)
        truncated_json_data = json_data[:30000]

        # --- Define Default Prompts ---
        # This prompt is now deprecated as we will use a single combined prompt.
        # default_rep_prompt = """..."""

        default_combined_prompt = """
You are an expert AI fitness coach. Your task is to analyze my performance of the following exercise: '{user_description}'.
Use the provided time-series JSON data of my body pose landmarks.

**Instructions:**
Analyze the provided data and return a single JSON object with the following structure. Do not include any text or markdown formatting outside of the JSON object itself.

{{
  "repetition_count": "<integer: Count the number of full repetitions. Be lenient in your counting.>",
  "form_correctness": "<string: Evaluate my form, symmetry, and range of motion for this specific exercise.>",
  "speed_pacing": "<string: Analyze the consistency and appropriateness of my movement speed.>",
  "corrective_feedback": "<string: Provide clear, actionable feedback on mistakes and how to improve my form.>",
  "overall_summary": "<string: A brief summary of my performance.>"
}}

**JSON Landmark Data:**
```json
{truncated_json_data}
```
"""
        # Get prompts from request or use defaults
        # rep_prompt_template is no longer used.
        form_prompt_template = request_data.get("form_prompt", default_combined_prompt)

        # --- PROMPT: Combined Analysis ---
        combined_prompt = form_prompt_template.format(
            user_description=user_description,
            truncated_json_data=truncated_json_data
        )
        
        response = gemini_model.generate_content(combined_prompt)
        
        # Clean up the response to ensure it's a valid JSON string
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
            print(f"[gemini] Warning: Could not parse combined analysis JSON. Error: {e}")
            analysis_json = {
                "error": "The model did not return a valid JSON object.",
                "raw_response": analysis_text,
                "repetition_count": 0 # Provide a default for the UI
            }
        
        print(f"[gemini] Combined analysis result: {analysis_json}")
        # Return the final combined JSON object
        return jsonify(analysis_json)

    except Exception as e:
        print(f"[gemini] Error during analysis: {e}")
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500
    finally:
        # --- Cleanup ---
        # Delete the temporary file after analysis is attempted
        if os.path.exists(session_file_path):
            try:
                os.remove(session_file_path)
                print(f"[cleanup] Deleted temporary file: {session_file_path}")
            except OSError as e:
                print(f"[cleanup] Error deleting file {session_file_path}: {e}")


@app.route("/save_json")
def save_json():
    # This endpoint is now deprecated as data is deleted after analysis.
    # You could adapt it to take a session_uid before it's analyzed.
    return jsonify({"error": "This endpoint is deprecated. Data is deleted after analysis."}), 410

if __name__ == "__main__":
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

    # Remove server-side camera initialization
    # The pose processor will start and wait for frames from the client
    print("[app] Starting pose processor thread.")
    threading.Thread(target=pose_processor_worker, daemon=True).start()

    print("[app] Flask server starting.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
