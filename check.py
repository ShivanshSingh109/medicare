# File: check_models.py
import google.generativeai as genai

# IMPORTANT: Replace with your key for this test
api_key = "AIzaSyCjYUIYAyZX0jaYGWewRorUEb9jOhjgaSg" 

try:
    genai.configure(api_key=api_key)
    print("Available models that support 'generateContent':")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"An error occurred: {e}")

import google.generativeai as genai

# IMPORTANT: Replace with your key for this test
api_key = "AIzaSyCjYUIYAyZX0jaYGWewRorUEb9jOhjgaSg" 

try:
    genai.configure(api_key=api_key)
    print("Available models that support 'generateContent':")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"An error occurred: {e}")