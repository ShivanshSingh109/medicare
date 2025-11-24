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

# Batching + token control constants
TARGET_MAX_PROMPT_CHARS = 6000      # soft cap per batch prompt
HARD_MAX_BATCH_FRAMES   = 200       # upper bound frames per batch
MIN_BATCH_FRAMES        = 40        # lower bound to avoid tiny batches
MAX_RETRIES             = 3
RETRY_BASE_DELAY        = 2.0  # seconds base delay for exponential backoff on 429

def _estimate_chars_for_frames(frames):
    # Quick heuristic: each landmark ~ 65 chars after rounding
    # Each frame keeps subset of landmarks.
    landmarks_per_frame = len(next(iter(frames[0]["landmarks"].values()))) if False else len(frames[0]["landmarks"])
    return len(frames) * landmarks_per_frame * 65

def _adaptive_batches(pose_data):
    # Start with HARD_MAX_BATCH_FRAMES and shrink if prompt too large.
    batches = []
    i = 0
    while i < len(pose_data):
        remaining = len(pose_data) - i
        size = min(HARD_MAX_BATCH_FRAMES, remaining)
        # Try shrink if char estimate too large
        while size > MIN_BATCH_FRAMES and _estimate_chars_for_frames(pose_data[i:i+size]) > TARGET_MAX_PROMPT_CHARS:
            size = int(size * 0.75)
        batch = pose_data[i:i+size]
        batches.append(batch)
        i += size
    print(f"[webcam] Adaptive batching produced {len(batches)} batches.")
    return batches

def _prepare_batch(batch):
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
        compact.append({"timestamp": frame.get("timestamp"), "landmarks": lm_compact})
    return compact

def _attempt_generate(parts):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return gemini_model.generate_content(parts)
        except Exception as e:
            if "429" not in str(e) and "quota" not in str(e).lower():
                raise
            delay = RETRY_BASE_DELAY * attempt
            print(f"[webcam] 429/quota retry {attempt}/{MAX_RETRIES} sleeping {delay:.1f}s")
            time.sleep(delay)
            last_err = e
    raise last_err

def _summarize_numeric_features(batch):
    # Produce condensed stats to reduce final consolidation token use
    # For each landmark produce min/max/avg for x,y,z (ignoring zeros with low visibility)
    stats = {}
    for frame in batch:
        for name, lm in frame["landmarks"].items():
            if name not in stats:
                stats[name] = {"x": [], "y": [], "z": []}
            # keep all values (already zeroed if invisible)
            stats[name]["x"].append(lm["x"])
            stats[name]["y"].append(lm["y"])
            stats[name]["z"].append(lm["z"])
    summary = {}
    for name, axes in stats.items():
        def agg(arr):
            if not arr:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": round(min(arr), 4),
                "max": round(max(arr), 4),
                "avg": round(sum(arr)/len(arr), 4),
            }
        summary[name] = {
            "x": agg(axes["x"]),
            "y": agg(axes["y"]),
            "z": agg(axes["z"]),
        }
    return summary

def _analyze_in_batches(pose_data, user_description, form_prompt_template):
    batches = _adaptive_batches(pose_data)
    batch_summaries = []

    for idx, batch in enumerate(batches, start=1):
        compact_batch = _prepare_batch(batch)

        # Convert batch to a compact plain-text format to reduce tokens
        # Frame lines: timestamp|landmark=x,y,z,vis;landmark2=...
        lines = []
        for frame in compact_batch:
            parts = []
            for lk, lv in frame["landmarks"].items():
                parts.append(f"{lk}={lv['x']:.4f},{lv['y']:.4f},{lv['z']:.4f},{lv['visibility']:.2f}")
            lines.append(f"{frame['timestamp']}|" + ";".join(parts))
        batch_text = "\n".join(lines)

        # Write plain text batch file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write(batch_text)
            temp_path = tf.name
        print(f"[webcam] Uploading batch {idx} text file ({len(batch)} frames)")
        uploaded = genai.upload_file(temp_path, mime_type="text/plain")
        os.unlink(temp_path)

        while uploaded.state.name == "PROCESSING":
            time.sleep(0.5)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED":
            raise RuntimeError("File upload failed for batch.")

        truncated_placeholder = "(Batch data provided as attached plain text file; each line = timestamp|landmark=x,y,z,visibility;...)"
        prompt = (
            form_prompt_template
            .replace("{user_description}", user_description)
            .replace("{truncated_json_data}", truncated_placeholder)
            + f"\nBatch index: {idx}; Frames: {len(batch)}"
        )

        summaries = _summarize_numeric_features(compact_batch)
        prompt += "\nCondensed landmark stats (min/max/avg x,y,z per landmark):\n"
        prompt += json.dumps(summaries)[:2500]

        response = _attempt_generate([prompt, uploaded])
        text = (response.text or "").strip()
        batch_summaries.append({"batch": idx, "raw": text})

    combined = "\n\n".join(f"Batch {b['batch']}:\n{b['raw']}" for b in batch_summaries)
    final_prompt = (
        "Consolidate the following batch analyses into ONE JSON with EXACTLY these keys: "
        "form_correctness, speed_pacing, corrective_feedback, overall_summary. "
        "Each value must be a single plain English string (no nested JSON). "
        "Return ONLY JSON.\n\n" + combined
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

        # Coerce non-string fields to strings to avoid [object Object] on client
        for k in ["form_correctness", "speed_pacing", "corrective_feedback", "overall_summary"]:
            if k in analysis_json and not isinstance(analysis_json[k], str):
                analysis_json[k] = json.dumps(analysis_json[k], ensure_ascii=False)

        analysis_json["total_frames"] = len(pose_data)
        analysis_json["batches_used"] = len(batch_meta)
        return jsonify(analysis_json)

    except Exception as e:
        # Add explicit token/size handling
        if any(k in str(e).lower() for k in ["token", "context", "quota", "429"]):
            return jsonify({"error": "Token/context limit or quota reached after retries."}), 429
        print(f"[webcam] Error: {e}")
        if "429" in str(e):
            return jsonify({"error": "Rate/resource limit reached after retries."}), 429
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500