import os
from flask import Flask, jsonify
from flask_cors import CORS

# -------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------
os.environ["MODEL_DIR"] = "/tmp/model"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# -------------------------------------------------------------
# Flask Initialization
# -------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------
# Import Blueprints (WebRTC only)
# -------------------------------------------------------------
try:
    print("🔍 Importing WebRTC Blueprint...")
    from routes.webrtc_routes import webrtc_bp
    app.register_blueprint(webrtc_bp)
    print("✅ WebRTC Blueprint registered successfully.")
except Exception as e:
    print(f"⚠️ Failed to import WebRTC Blueprint: {e}")

# -------------------------------------------------------------
# ✅ Enable CORS *after* registering all blueprints
# -------------------------------------------------------------
CORS(
    app,
    origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://proctorvision-client.vercel.app",
        "https://proctorvision-webrtc-production.up.railway.app",
    ],
    supports_credentials=True,
)

# -------------------------------------------------------------
# Root & Health Check Route
# -------------------------------------------------------------
@app.route("/")
def home():
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify({
        "status": "ok",
        "message": "✅ ProctorVision WebRTC Backend Running",
        "available_routes": routes
    })

# -------------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway default port
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
