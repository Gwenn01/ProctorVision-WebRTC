# routes/xirsys_routes.py
import base64
import os
import requests
from flask import Blueprint, jsonify

xirsys_bp = Blueprint("xirsys_bp", __name__)

@xirsys_bp.route("/get-turn", methods=["GET"])
def get_turn_credentials():
    """Fetch dynamic TURN credentials from Xirsys"""
    XIRSYS_USER = os.getenv("XIRSYS_USER", "chatgpt3c")
    XIRSYS_SECRET = os.getenv("XIRSYS_SECRET", "cae385bc-b265-11f0-9f51-0242ac130002")
    XIRSYS_CHANNEL = os.getenv("XIRSYS_CHANNEL", "MyFirstApp")

    if not XIRSYS_USER or not XIRSYS_SECRET or not XIRSYS_CHANNEL:
        return jsonify({"error": "Missing Xirsys credentials or channel"}), 500

    # Build Authorization header
    auth_str = f"{XIRSYS_USER}:{XIRSYS_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Accept": "application/json"
    }

    url = f"https://global.xirsys.net/_turn/{XIRSYS_CHANNEL}"
    print(f"[TURN] Requesting credentials from {url}")

    try:
        resp = requests.put(url, headers=headers, timeout=10)
        data = resp.json()

        # Handle unexpected responses cleanly
        if data.get("s") != "ok":
            print(f"[TURN ERROR] Xirsys returned error: {data}")
            return jsonify({"error": "Failed to fetch TURN credentials", "details": data}), 500

        return jsonify(data)
    except requests.exceptions.RequestException as e:
        print(f"[TURN REQUEST FAILED] {e}")
        return jsonify({"error": "Request to Xirsys failed", "details": str(e)}), 500
