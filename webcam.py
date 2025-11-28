import os
import json
import uuid
import time
from flask import render_template, jsonify, request, send_from_directory, session
import google.generativeai as genai
import math

JSON_DATA_FOLDER = "analysis_data"
gemini_model = None

# Import profile loader
from auth import get_user_profile

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

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

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

def _calculate_angle(p1, p2, p3):
    """Calculate angle at p2 given three points"""
    if any(p.get("visibility", 0) < 0.5 for p in [p1, p2, p3]):
        return None
    
    v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 * mag2 == 0:
        return None
    
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))

def _downsample_series(series, max_points=5000):
    """Downsample a time series to max_points while preserving pattern"""
    if len(series) <= max_points:
        return series
    
    # Take evenly spaced samples
    step = len(series) / max_points
    return [series[int(i * step)] for i in range(max_points)]

def _extract_movement_features(pose_data, max_samples_per_joint=5000):
    """Extract movement features with full angle time series"""
    
    if not pose_data:
        return {}
    
    # Define joint angle configurations
    angle_configs = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
        "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    }
    
    # Collect angles over time (full series)
    angle_series = {name: [] for name in angle_configs}
    
    # Collect position changes for movement tracking
    position_series = {lm: {"x": [], "y": []} for lm in [
        "left_wrist", "right_wrist", "left_ankle", "right_ankle",
        "nose", "left_hip", "right_hip"
    ]}
    
    timestamps = []
    
    for frame in pose_data:
        lm = frame.get("landmarks", {})
        timestamps.append(frame.get("timestamp", ""))
        
        # Calculate joint angles for this frame
        for angle_name, (p1_name, p2_name, p3_name) in angle_configs.items():
            p1 = lm.get(p1_name, {})
            p2 = lm.get(p2_name, {})
            p3 = lm.get(p3_name, {})
            angle = _calculate_angle(p1, p2, p3)
            if angle is not None:
                angle_series[angle_name].append(round(angle, 1))
            else:
                angle_series[angle_name].append(None)
        
        # Track positions
        for lm_name in position_series:
            if lm_name in lm and lm[lm_name].get("visibility", 0) > 0.5:
                position_series[lm_name]["x"].append(round(lm[lm_name]["x"], 4))
                position_series[lm_name]["y"].append(round(lm[lm_name]["y"], 4))
            else:
                position_series[lm_name]["x"].append(None)
                position_series[lm_name]["y"].append(None)
    
    # Process angle series - remove None values and downsample if needed
    processed_angles = {}
    for name, values in angle_series.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            downsampled = _downsample_series(valid_values, max_samples_per_joint)
            processed_angles[name] = {
                "values": downsampled,
                "count": len(valid_values),
                "min": round(min(valid_values), 1),
                "max": round(max(valid_values), 1),
                "range": round(max(valid_values) - min(valid_values), 1)
            }
    
    # Process position series
    processed_positions = {}
    for lm_name, coords in position_series.items():
        valid_x = [v for v in coords["x"] if v is not None]
        valid_y = [v for v in coords["y"] if v is not None]
        if valid_x and valid_y:
            downsampled_x = _downsample_series(valid_x, max_samples_per_joint)
            downsampled_y = _downsample_series(valid_y, max_samples_per_joint)
            processed_positions[lm_name] = {
                "x_values": downsampled_x,
                "y_values": downsampled_y,
                "x_range": round(max(valid_x) - min(valid_x), 4),
                "y_range": round(max(valid_y) - min(valid_y), 4)
            }
    
    # Estimate repetitions by counting peaks
    def count_reps(values, threshold=15):
        if len(values) < 5:
            return 0
        peaks = 0
        increasing = False
        for i in range(1, len(values)):
            if values[i] > values[i-1] + threshold and not increasing:
                increasing = True
            elif values[i] < values[i-1] - threshold and increasing:
                peaks += 1
                increasing = False
        return peaks
    
    rep_estimates = {}
    for name, data in processed_angles.items():
        reps = count_reps(data["values"])
        if reps > 0:
            rep_estimates[name] = reps
    
    # Duration estimation
    duration_seconds = None
    if len(timestamps) >= 2:
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            duration_seconds = (t2 - t1).total_seconds()
        except:
            duration_seconds = len(timestamps) / 30
    
    return {
        "total_frames": len(pose_data),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
        "samples_per_joint": max_samples_per_joint,
        "joint_angles": processed_angles,
        "body_positions": processed_positions,
        "estimated_reps": rep_estimates,
        "frames_per_second": round(len(pose_data) / duration_seconds, 1) if duration_seconds and duration_seconds > 0 else None
    }

def _chunk_data_for_llm(features, max_tokens_estimate=8000):
    """
    Split features into chunks if needed to avoid token limits.
    """
    def estimate_tokens(data):
        json_str = json.dumps(data)
        return len(json_str) // 4
    
    total_tokens = estimate_tokens(features)
    
    if total_tokens <= max_tokens_estimate:
        return [features]
    
    chunks = []
    joint_names = list(features.get("joint_angles", {}).keys())
    position_names = list(features.get("body_positions", {}).keys())
    
    avg_tokens_per_joint = total_tokens // (len(joint_names) + len(position_names) + 1)
    joints_per_chunk = max(1, max_tokens_estimate // avg_tokens_per_joint - 2)
    
    base_info = {
        "total_frames": features.get("total_frames"),
        "duration_seconds": features.get("duration_seconds"),
        "samples_per_joint": features.get("samples_per_joint"),
        "frames_per_second": features.get("frames_per_second"),
        "estimated_reps": features.get("estimated_reps", {})
    }
    
    for i in range(0, len(joint_names), joints_per_chunk):
        chunk_joints = joint_names[i:i + joints_per_chunk]
        chunk = base_info.copy()
        chunk["joint_angles"] = {k: features["joint_angles"][k] for k in chunk_joints}
        chunk["chunk_info"] = f"Joints: {', '.join(chunk_joints)}"
        chunks.append(chunk)
    
    if position_names:
        position_chunk = base_info.copy()
        position_chunk["body_positions"] = features.get("body_positions", {})
        position_chunk["chunk_info"] = f"Body positions: {', '.join(position_names)}"
        chunks.append(position_chunk)
    
    return chunks

def _create_analysis_prompt(features, user_description, patient_profile=None, chunk_index=None, total_chunks=None):
    """Create a prompt with full angle time series data and patient context"""
    
    chunk_info = ""
    if chunk_index is not None and total_chunks is not None:
        chunk_info = f"\n(This is part {chunk_index + 1} of {total_chunks} - analyze this data portion)\n"
    
    # Build patient context
    patient_context = ""
    if patient_profile:
        condition_map = {
            "back_pain": "Back Pain",
            "neck_pain": "Neck Pain", 
            "shoulder_injury": "Shoulder Injury",
            "knee_injury": "Knee Injury",
            "hip_pain": "Hip Pain",
            "post_surgery": "Post-Surgery Recovery",
            "sports_injury": "Sports Injury",
            "arthritis": "Arthritis",
            "general_mobility": "General Mobility Issues",
            "other": "Other Condition"
        }
        
        mobility_map = {
            "fully_mobile": "Fully mobile",
            "slightly_limited": "Slightly limited mobility",
            "moderately_limited": "Moderately limited mobility",
            "severely_limited": "Severely limited mobility",
            "very_limited": "Very limited mobility"
        }
        
        experience_map = {
            "beginner": "Beginner",
            "some_experience": "Some exercise experience",
            "regular": "Regular exerciser",
            "active": "Very active",
            "athlete": "Athlete level"
        }
        
        patient_context = f"""
## PATIENT PROFILE (Consider this when giving feedback):
- Name: {patient_profile.get('username', 'Patient')}
- Age: {patient_profile.get('age', 'Unknown')}
- Gender: {patient_profile.get('gender', 'Unknown')}
- Primary Condition: {condition_map.get(patient_profile.get('primary_condition', ''), 'Unknown')}
- Current Pain Level: {patient_profile.get('pain_level', 'Unknown')}/10
- Mobility: {mobility_map.get(patient_profile.get('mobility_level', ''), 'Unknown')}
- Exercise Experience: {experience_map.get(patient_profile.get('exercise_experience', ''), 'Unknown')}
- Condition Details: {patient_profile.get('condition_description', 'Not provided')}
- Goals: {patient_profile.get('goals', 'Not specified')}
- Medical History: {', '.join(patient_profile.get('medical_history', [])) or 'None reported'}

IMPORTANT: Tailor your feedback to this patient's specific condition, pain level, and experience. Be mindful of their limitations and goals.
"""
    
    prompt = f"""You are an expert AI physiotherapy coach analyzing the exercise: '{user_description}'
{chunk_info}
{patient_context}
## CRITICAL INSTRUCTIONS:
- ALWAYS provide specific, actionable feedback based on the data provided
- NEVER say you need more data, cannot analyze, or ask for additional information
- If movement is minimal, provide feedback on what the user SHOULD be doing for this exercise
- If you think the user is not actually performing the exercise, clearly say so in your feedback
- Be encouraging but specific about improvements
- DO NOT mention specific degree values, angles, or rep counts in your response
- Keep each response field to 1-2 sentences maximum
- Use simple, conversational language
- Consider the patient's condition and limitations when giving advice

## Recording Info:
- Total frames: {features.get('total_frames', 'N/A')}
- Duration: {features.get('duration_seconds', 'N/A')} seconds

## Joint Angle Time Series (degrees over time):
"""
    
    for joint, data in features.get("joint_angles", {}).items():
        values_str = ", ".join(str(v) for v in data["values"])
        prompt += f"\n### {joint}:\n"
        prompt += f"- Range: {data['min']}° to {data['max']}° (movement range: {data['range']}°)\n"
        prompt += f"- Values: [{values_str}]\n"
    
    if features.get("body_positions"):
        prompt += "\n## Body Position Tracking (normalized 0-1):\n"
        for part, data in features.get("body_positions", {}).items():
            x_str = ", ".join(str(v) for v in data["x_values"][:30]) + ("..." if len(data["x_values"]) > 30 else "")
            y_str = ", ".join(str(v) for v in data["y_values"][:30]) + ("..." if len(data["y_values"]) > 30 else "")
            prompt += f"\n### {part}:\n"
            prompt += f"- Movement: X={data['x_range']}, Y={data['y_range']}\n"
            prompt += f"- X: [{x_str}]\n"
            prompt += f"- Y: [{y_str}]\n"
    
    if features.get("estimated_reps"):
        prompt += "\n## Detected Repetitions:\n"
        for joint, reps in features["estimated_reps"].items():
            prompt += f"- {joint}: {reps} reps\n"
    
    prompt += f"""
## Your Analysis Task for '{user_description}':
Provide brief, natural feedback (1-2 sentences each, NO technical numbers):

1. **Form Correctness** - Is the form good? What looks right or needs work?
2. **Consistency** - Are the movements smooth and steady?
3. **Speed/Pacing** - Is the speed appropriate?
4. **Corrective Feedback** - 1-2 quick tips to improve
5. **Overall Summary** - One encouraging sentence

Return ONLY this JSON (no markdown, no numbers, no degree symbols):
{{"form_correctness": "<brief natural feedback>", "consistency": "<brief feedback>", "speed_pacing": "<brief feedback>", "corrective_feedback": "<1-2 quick tips>", "overall_summary": "<one encouraging sentence>"}}
"""
    return prompt

def analyze_exercise():
    if gemini_model is None:
        return jsonify({"error": "Gemini model not initialized."}), 500

    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Invalid request."}), 400

    pose_data = request_data.get("pose_data")
    user_description = (request_data.get("description") or "an exercise").strip()
    
    # Try to get patient_profile from request_data first (for background thread)
    patient_profile = request_data.get("patient_profile")
    if not patient_profile and 'user_id' in session:
        from auth import get_user_profile
        patient_profile = get_user_profile(session['user_id'])
    
    # Increased max samples to 5000
    max_samples = min(int(request_data.get("max_samples", 5000)), 10000)

    if not pose_data:
        return jsonify({"error": "No pose_data provided."}), 400

    try:
        filename = f"{uuid.uuid4()}.json"
        filepath = os.path.join(JSON_DATA_FOLDER, filename)
        os.makedirs(JSON_DATA_FOLDER, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(pose_data, f, indent=0)
        print(f"[webcam] Saved full pose data ({len(pose_data)} frames) to '{filepath}'")

        features = _extract_movement_features(pose_data, max_samples_per_joint=max_samples)
        print(f"[webcam] Extracted features for {len(features.get('joint_angles', {}))} joints")
        
        chunks = _chunk_data_for_llm(features)
        print(f"[webcam] Data split into {len(chunks)} chunk(s)")
        
        all_responses = []
        
        for i, chunk in enumerate(chunks):
            prompt = _create_analysis_prompt(
                chunk, 
                user_description,
                patient_profile=patient_profile,
                chunk_index=i if len(chunks) > 1 else None,
                total_chunks=len(chunks) if len(chunks) > 1 else None
            )
            print(f"[webcam] Chunk {i+1} prompt length: {len(prompt)} chars")
            
            response = _attempt_generate([prompt])
            response_text = (response.text or "").strip()
            all_responses.append(response_text)
        
        if len(all_responses) == 1:
            final_response = all_responses[0]
        else:
            combine_prompt = f"""Combine these {len(all_responses)} partial analyses for '{user_description}' into one cohesive response.

{chr(10).join(all_responses)}

Return ONLY this JSON:
{{"form_correctness": "<combined>", "consistency": "<combined>", "speed_pacing": "<combined>", "corrective_feedback": "<combined tips>", "overall_summary": "<one sentence>"}}
"""
            combine_response = _attempt_generate([combine_prompt])
            final_response = (combine_response.text or "").strip()

        try:
            json_start = final_response.find('{')
            json_end = final_response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response.")
            analysis_json = json.loads(final_response[json_start:json_end])
        except Exception as e:
            print(f"[webcam] JSON parse warning: {e}")
            analysis_json = {
                "error": "Model did not return valid JSON.",
                "raw_response": final_response
            }

        for k in ["form_correctness", "consistency", "speed_pacing", "corrective_feedback", "overall_summary"]:
            if k in analysis_json and not isinstance(analysis_json[k], str):
                analysis_json[k] = json.dumps(analysis_json[k], ensure_ascii=False)

        analysis_json["total_frames"] = len(pose_data)
        analysis_json["chunks_used"] = len(chunks)
        analysis_json["extracted_features"] = features
        analysis_json["description"] = user_description  # Add exercise name

        # Save analysis result to a file for later reference
        analysis_filename = f"analysis_{uuid.uuid4().hex}.json"
        analysis_filepath = os.path.join(JSON_DATA_FOLDER, analysis_filename)
        with open(analysis_filepath, "w") as f:
            json.dump(analysis_json, f, indent=2)

        # Return the filename so it can be linked
        return jsonify({**analysis_json, "analysis_file": analysis_filename})

    except Exception as e:
        if any(k in str(e).lower() for k in ["token", "context", "quota", "429"]):
            return jsonify({"error": "Token/context limit or quota reached after retries."}), 429
        print(f"[webcam] Error: {e}")
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500

def run_analysis_logic(pose_data, user_description, patient_profile=None, max_samples=5000):
    if gemini_model is None:
        raise ValueError("Gemini model not initialized.")

    if not pose_data:
        raise ValueError("No pose_data provided.")

    # Save pose data to file
    filename = f"{uuid.uuid4()}.json"
    filepath = os.path.join(JSON_DATA_FOLDER, filename)
    os.makedirs(JSON_DATA_FOLDER, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(pose_data, f, indent=0)
    print(f"[webcam] Saved full pose data ({len(pose_data)} frames) to '{filepath}'")

    features = _extract_movement_features(pose_data, max_samples_per_joint=max_samples)
    print(f"[webcam] Extracted features for {len(features.get('joint_angles', {}))} joints")
    
    chunks = _chunk_data_for_llm(features)
    print(f"[webcam] Data split into {len(chunks)} chunk(s)")
    
    all_responses = []
    
    for i, chunk in enumerate(chunks):
        prompt = _create_analysis_prompt(
            chunk, 
            user_description,
            patient_profile=patient_profile,
            chunk_index=i if len(chunks) > 1 else None,
            total_chunks=len(chunks) if len(chunks) > 1 else None
        )
        print(f"[webcam] Chunk {i+1} prompt length: {len(prompt)} chars")
        
        response = _attempt_generate([prompt])
        response_text = (response.text or "").strip()
        all_responses.append(response_text)
    
    if len(all_responses) == 1:
        final_response = all_responses[0]
    else:
        combine_prompt = f"""Combine these {len(all_responses)} partial analyses for '{user_description}' into one cohesive response.

{chr(10).join(all_responses)}

Return ONLY this JSON:
{{"form_correctness": "<combined>", "consistency": "<combined>", "speed_pacing": "<combined>", "corrective_feedback": "<combined tips>", "overall_summary": "<one sentence>"}}
"""
        combine_response = _attempt_generate([combine_prompt])
        final_response = (combine_response.text or "").strip()

    try:
        json_start = final_response.find('{')
        json_end = final_response.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in response.")
        analysis_json = json.loads(final_response[json_start:json_end])
    except Exception as e:
        print(f"[webcam] JSON parse warning: {e}")
        analysis_json = {
            "error": "Model did not return valid JSON.",
            "raw_response": final_response
        }

    for k in ["form_correctness", "consistency", "speed_pacing", "corrective_feedback", "overall_summary"]:
        if k in analysis_json and not isinstance(analysis_json[k], str):
            analysis_json[k] = json.dumps(analysis_json[k], ensure_ascii=False)

    analysis_json["total_frames"] = len(pose_data)
    analysis_json["chunks_used"] = len(chunks)
    analysis_json["extracted_features"] = features
    analysis_json["description"] = user_description  # Add exercise name

    # Save analysis result to a file for later reference
    analysis_filename = f"analysis_{uuid.uuid4().hex}.json"
    analysis_filepath = os.path.join(JSON_DATA_FOLDER, analysis_filename)
    with open(analysis_filepath, "w") as f:
        json.dump(analysis_json, f, indent=2)

    # Return analysis result
    return analysis_json, analysis_filename