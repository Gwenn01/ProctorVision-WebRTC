import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# -------------------------------------------------------------
# ✅ Environment Configuration
# -------------------------------------------------------------
os.environ["MODEL_DIR"] = "/tmp/model"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TensorFlow INFO/WARN logs
os.environ["GLOG_minloglevel"] = "2"       # Hide Mediapipe logs
os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# -------------------------------------------------------------
# ✅ Flask Initialization
# -------------------------------------------------------------
app = Flask(__name__)

# ✅ CORS Configuration — Allow both local and production frontends
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://proctorvision-client.vercel.app",
                "https://proctorvision-webrtc-production.up.railway.app",
                "https://*.vercel.app"
            ]
        }
    },
    supports_credentials=True,
)

# -------------------------------------------------------------
# ✅ Import WebRTC Blueprint
# -------------------------------------------------------------
try:
    print("🔍 Importing WebRTC Blueprint...")
    from routes.webrtc_routes import webrtc_bp
    from routes.xirsys_routes import xirsys_bp
    app.register_blueprint(webrtc_bp)
    app.register_blueprint(xirsys_bp, url_prefix="/api")
    print("✅ WebRTC Blueprint registered successfully.")
except Exception as e:
    print(f"⚠️ Failed to import WebRTC Blueprint: {e}")

# -------------------------------------------------------------
# ✅ Global CORS Header Injection (Fallback)
# -------------------------------------------------------------
@app.after_request
def global_cors(response):
    """Ensures every response contains CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = "https://proctor-vision-client.vercel.app"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

# -------------------------------------------------------------
# ✅ Root & Health Check
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
# ✅ Debug Route — Show incoming origin (for CORS diagnostics)
# -------------------------------------------------------------
@app.before_request
def debug_origin():
    print(">>> Incoming request from origin:", request.origin or "Unknown")

# -------------------------------------------------------------
# ✅ Startup Log — Show all available routes
# -------------------------------------------------------------
print("✅ WebRTC server boot completed — available routes:")
try:
    from routes.webrtc_routes import webrtc_bp  # Ensures registration before listing
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
except Exception:
    print("⚠️ Could not list routes (may not be registered yet).")

# -------------------------------------------------------------
# ✅ Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"🚀 Starting WebRTC server on port {port} (debug={debug})...")
    app.run(host="0.0.0.0", port=port, debug=debug)
