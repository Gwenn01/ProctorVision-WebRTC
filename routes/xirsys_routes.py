# routes/xirsys_routes.py
import base64, os, requests
from flask import Blueprint, jsonify

xirsys_bp = Blueprint("xirsys_bp", __name__)

@xirsys_bp.route("/get-turn", methods=["GET"])
def get_turn_credentials():
    """Fetch dynamic TURN credentials from Xirsys"""
    XIRSYS_USER = os.getenv("XIRSYS_USER", "Gwenn01")
    XIRSYS_SECRET = os.getenv("XIRSYS_SECRET", "944c0d5e-acd8-11f0-b97b-0242ac130002")
    XIRSYS_CHANNEL = os.getenv("XIRSYS_CHANNEL", "MyFirstApp")

    if not XIRSYS_USER or not XIRSYS_SECRET:
        return jsonify({"error": "Missing Xirsys credentials"}), 500

    auth_str = f"{XIRSYS_USER}:{XIRSYS_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {auth_b64}"}
    url = f"https://global.xirsys.net/_turn/{XIRSYS_CHANNEL}"

    try:
        print(f"[TURN] Requesting TURN credentials from {url}")
        resp = requests.put(url, headers=headers)
        return jsonify(resp.json())
    except Exception as e:
        print(f"[TURN ERROR] {e}")
        return jsonify({"error": str(e)}), 500
