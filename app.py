import os
from flask import Flask, jsonify

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
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
