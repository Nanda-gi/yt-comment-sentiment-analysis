import pytest
import requests
import json
from flask import jsonify

BASE_URL = "http://127.0.0.1:5000"  # Replace with your deployed URL if needed

def test_predict_endpoint():
    data = {
        "comments": ["This is a great product!", "Not worth the money.", "It's okay."]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
@app.route('/health', methods=['GET'])

def health_check():
    return jsonify({"status": "ok"}), 200
