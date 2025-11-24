import os
import json
import uuid
import tempfile
import time
from flask import render_template, jsonify, request, send_from_directory
import google.generativeai as genai
import math

JSON_DATA_FOLDER = "analysis_data"
gemini_model = None

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in your .env file.")
    genai.configure(api_key=api_key)
    global gemini_model
    gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

def index():
    return render_template("index.html")

def serve_model(filename):
    return send_from_directory(".", filename)

# Hard‑coded batching constants
BATCH_SIZE = 200        # number of frames per batch
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds base delay for exponential backoff on 429

def _prepare_batch(batch):
    # Keep all frames, optionally lightly compact numeric precision
    compact = []
    for frame in batch:
        lm_compact = {}
        for k, v in frame["landmarks"].items():
            lm_compact[k] = {
                "x": round(v.get("x", 0), 4),
                "y": round(v.get("y", 0), 4),
                "z": round(v.get("z", 0), 4),
                "visibility": round(v.get("visibility", 0), 4),
            }
        compact.append({
            "timestamp": frame.get("timestamp"),
            "landmarks": lm_compact
        })
    return compact

def _attempt_generate(parts):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return gemini_model.generate_content(parts)
        except Exception as e:
            msg = str(e)
            if "429" not in msg:
                raise
            delay = RETRY_BASE_DELAY * attempt
            print(f"[webcam] 429 retry {attempt}/{MAX_RETRIES} sleeping {delay:.1f}s")
            time.sleep(delay)
            last_err = e
    raise last_err

def _analyze_in_batches(pose_data, user_description, form_prompt_template):
    # Split ALL frames into batches (no dropping)
    batches = [
        pose_data[i:i + BATCH_SIZE] for i in range(0, len(pose_data), BATCH_SIZE)
    ]
    print(f"[webcam] Using all {len(pose_data)} frames across {len(batches)} batches (size={BATCH_SIZE})")

    batch_summaries = []
    for idx, batch in enumerate(batches, start=1):
        compact_batch = _prepare_batch(batch)
        # Limit JSON string length defensively (won't drop frames, just truncates textual representation in prompt)
        json_str = json.dumps(compact_batch)
        truncated_json = json_str[:18000]  # prompt safety truncation; frames still sent logically per batch
        prompt = (
            form_prompt_template
            .replace("{user_description}", user_description)
            .replace("{truncated_json_data}", truncated_json)
        )
        print(f"[webcam] Sending batch {idx}/{len(batches)} with {len(batch)} frames")
        response = _attempt_generate([prompt])
        text = (response.text or "").strip()
        batch_summaries.append({"batch": idx, "raw": text})

    # Consolidation step
    combined_texts = "\n\n".join(
        f"Batch {b['batch']}:\n{b['raw']}" for b in batch_summaries
    )
    final_prompt = (
        "You are an expert AI fitness coach. Consolidate the per-batch analyses below "
        "into ONE final JSON object with keys: form_correctness, speed_pacing, "
        "corrective_feedback, overall_summary. Return ONLY JSON.\n\n"
        f"{combined_texts}"
    )
    final_resp = _attempt_generate([final_prompt])
    return final_resp.text, batch_summaries

def analyze_exercise():
    if gemini_model is None:
        return jsonify({"error": "Gemini model not initialized."}), 500

    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Invalid request."}), 400

    pose_data = request_data.get("pose_data")
    user_description = (request_data.get("description") or "an exercise").strip()
    form_prompt_template = request_data.get("form_prompt")

    if not pose_data:
        return jsonify({"error": "No pose_data provided."}), 400
    if not form_prompt_template:
        form_prompt_template = "Analyze the exercise '{user_description}' using this data:\n{truncated_json_data}"

    try:
        # Save full raw frames
        filename = f"{uuid.uuid4()}.json"
        filepath = os.path.join(JSON_DATA_FOLDER, filename)
        os.makedirs(JSON_DATA_FOLDER, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(pose_data, f, indent=0)
        print(f"[webcam] Saved full pose data ({len(pose_data)} frames) to '{filepath}'")

        final_text, batch_meta = _analyze_in_batches(
            pose_data, user_description, form_prompt_template
        )

        try:
            s = final_text
            json_start = s.find('{')
            json_end = s.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in final response.")
            analysis_json = json.loads(s[json_start:json_end])
        except Exception as e:
            print(f"[webcam] Final JSON parse warning: {e}")
            analysis_json = {
                "error": "Model did not return valid JSON.",
                "raw_response": final_text
            }

        analysis_json["total_frames"] = len(pose_data)
        analysis_json["batches_used"] = len(batch_meta)
        return jsonify(analysis_json)

    except Exception as e:
        print(f"[webcam] Error: {e}")
        if "429" in str(e):
            return jsonify({"error": "Rate/resource limit reached after retries."}), 429
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500