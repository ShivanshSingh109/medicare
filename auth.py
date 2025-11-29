import os
import json
import hashlib
import re
from functools import wraps
from flask import render_template, request, jsonify, session, redirect, url_for

USERS_FILE = "users.json"
PROFILES_FILE = "patient_profiles.json"

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_profiles():
    """Load patient profiles from JSON file"""
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)

def save_profiles(profiles):
    """Save patient profiles to JSON file"""
    with open(PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

def get_user_profile(email):
    """Get profile for a specific user"""
    profiles = load_profiles()
    return profiles.get(email)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Valid"

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def profile_required(f):
    """Decorator to require completed profile"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        
        # Check if profile exists
        profile = get_user_profile(session['user_id'])
        if not profile:
            return redirect(url_for('profile_setup_page'))
        
        return f(*args, **kwargs)
    return decorated_function

def login_page():
    """Render login page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template("login.html")

def register_page():
    """Render registration page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template("register.html")

def profile_setup_page():
    """Render profile setup page"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    # If profile already exists, redirect to main page
    profile = get_user_profile(session['user_id'])
    if profile:
        return redirect(url_for('exercise_dashboard'))
    
    return render_template("profile_setup.html")

def save_profile():
    """Handle profile form submission"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['age', 'gender', 'primary_condition', 'pain_level', 
                       'mobility_level', 'condition_description', 'exercise_experience']
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Validate age
    try:
        age = int(data['age'])
        if age < 1 or age > 120:
            return jsonify({"error": "Invalid age"}), 400
    except:
        return jsonify({"error": "Invalid age"}), 400
    
    # Validate pain level
    try:
        pain_level = int(data['pain_level'])
        if pain_level < 0 or pain_level > 10:
            return jsonify({"error": "Invalid pain level"}), 400
    except:
        return jsonify({"error": "Invalid pain level"}), 400
    
    # Get user info
    users = load_users()
    user_id = session['user_id']
    user_data = users.get(user_id, {})
    
    # Create profile
    profile = {
        "email": user_id,
        "username": user_data.get('username', session.get('username', '')),
        "age": age,
        "gender": data['gender'],
        "primary_condition": data['primary_condition'],
        "pain_level": pain_level,
        "mobility_level": data['mobility_level'],
        "medical_history": data.get('medical_history', []),
        "condition_description": data['condition_description'],
        "goals": data.get('goals', ''),
        "exercise_experience": data['exercise_experience'],
        "created_at": __import__('datetime').datetime.now().isoformat(),
        "updated_at": __import__('datetime').datetime.now().isoformat()
    }
    
    # Save profile
    profiles = load_profiles()
    profiles[user_id] = profile
    save_profiles(profiles)
    
    return jsonify({"message": "Profile saved successfully"}), 200

def update_profile():
    """Handle profile update"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    profiles = load_profiles()
    user_id = session['user_id']
    
    if user_id not in profiles:
        return jsonify({"error": "Profile not found"}), 404
    
    # Update fields
    profile = profiles[user_id]
    updatable_fields = ['age', 'gender', 'primary_condition', 'pain_level', 
                        'mobility_level', 'medical_history', 'condition_description', 
                        'goals', 'exercise_experience']
    
    for field in updatable_fields:
        if field in data:
            profile[field] = data[field]
    
    profile['updated_at'] = __import__('datetime').datetime.now().isoformat()
    
    save_profiles(profiles)
    
    return jsonify({"message": "Profile updated successfully"}), 200

def get_profile():
    """Get current user's profile"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    profile = get_user_profile(session['user_id'])
    if not profile:
        return jsonify({"error": "Profile not found", "has_profile": False}), 404
    
    return jsonify({"profile": profile, "has_profile": True}), 200

def register():
    """Handle user registration"""
    data = request.get_json()
    
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')
    
    # Validation
    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters long"}), 400
    
    if not email or not validate_email(email):
        return jsonify({"error": "Invalid email format"}), 400
    
    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    
    is_valid, msg = validate_password(password)
    if not is_valid:
        return jsonify({"error": msg}), 400
    
    # Check if user already exists
    users = load_users()
    
    if email in users:
        return jsonify({"error": "Email already registered"}), 400
    
    for user_data in users.values():
        if user_data['username'].lower() == username.lower():
            return jsonify({"error": "Username already taken"}), 400
    
    # Create new user
    user_id = email
    users[user_id] = {
        "username": username,
        "email": email,
        "password": hash_password(password)
    }
    
    save_users(users)
    
    return jsonify({"message": "Registration successful", "username": username}), 201

def login():
    """Handle user login"""
    data = request.get_json()
    
    identifier = data.get('identifier', '').strip().lower()  # username or email
    password = data.get('password', '')
    
    if not identifier or not password:
        return jsonify({"error": "Please provide username/email and password"}), 400
    
    users = load_users()
    
    # Find user by email or username
    user_data = None
    user_id = None
    
    if identifier in users:
        user_id = identifier
        user_data = users[identifier]
    else:
        for uid, udata in users.items():
            if udata['username'].lower() == identifier:
                user_id = uid
                user_data = udata
                break
    
    if not user_data:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Verify password
    if user_data['password'] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Set session
    session['user_id'] = user_id
    session['username'] = user_data['username']
    
    # Check if profile exists
    profile = get_user_profile(user_id)
    has_profile = profile is not None
    
    return jsonify({
        "message": "Login successful", 
        "username": user_data['username'],
        "has_profile": has_profile,
        "redirect": "/exercise_dashboard" if has_profile else "/profile-setup"
    }), 200

def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('login_page'))

def get_current_user():
    """Get current logged-in user info"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    profile = get_user_profile(session['user_id'])
    
    return jsonify({
        "username": session.get('username'),
        "email": session.get('user_id'),
        "has_profile": profile is not None
    }), 200