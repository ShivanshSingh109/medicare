import os
import json
import hashlib
import re
from functools import wraps
from flask import render_template, request, jsonify, session, redirect, url_for

USERS_FILE = "users.json"

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
    
    return jsonify({"message": "Login successful", "username": user_data['username']}), 200

def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('login_page'))

def get_current_user():
    """Get current logged-in user info"""
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    return jsonify({
        "username": session.get('username'),
        "email": session.get('user_id')
    }), 200