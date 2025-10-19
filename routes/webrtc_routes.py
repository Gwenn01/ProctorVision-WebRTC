import asyncio, time, traceback, os, threading, base64, cv2, numpy as np, mediapipe as mp
from collections import defaultdict, deque
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from flask import Blueprint, request, jsonify
from database.connection import get_db_connection  # ✅ direct DB connection
from flask_cors import CORS

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
webrtc_bp = Blueprint("webrtc", __name__)
CORS(
    webrtc_bp,
    origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://proctorvision-client.vercel.app",
        "https://proctorvision-webrtc-production.up.railway.app",
    ],
    supports_credentials=True,
)

SUMMARY_EVERY_S = float(os.getenv("PROCTOR_SUMMARY_EVERY_S", "1.0"))
RECV_TIMEOUT_S = float(os.getenv("PROCTOR_RECV_TIMEOUT_S", "5.0"))
HEARTBEAT_S = float(os.getenv("PROCTOR_HEARTBEAT_S", "10.0"))

# ----------------------------------------------------------------------
# LOGGING UTIL
# ----------------------------------------------------------------------
def log(event, sid="-", eid="-", **kv):
    tail = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{event}] sid={sid} eid={eid} {tail}".strip(), flush=True)

# ----------------------------------------------------------------------
# GLOBAL STATE
# ----------------------------------------------------------------------
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()
pcs = set()
last_warning = defaultdict(lambda: {"warning": "Looking Forward", "at": 0})
last_capture = defaultdict(lambda: {"label": None, "at": 0})
last_metrics = defaultdict(lambda: {
    "yaw": None, "pitch": None, "dx": None, "dy": None,
    "fps": None, "label": "n/a", "at": 0
})

# ----------------------------------------------------------------------
# MEDIAPIPE SETUP
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ----------------------------------------------------------------------
# DETECTOR CLASS
# ----------------------------------------------------------------------
IDX_NOSE, IDX_CHIN, IDX_LE, IDX_RE, IDX_LM, IDX_RM = 1, 152, 263, 33, 291, 61
MODEL_3D = np.array([
    [0.0,   0.0,   0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3,  32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9,  -28.9, -24.1],
], dtype=np.float32)

def _landmarks_to_pts(lms, w, h):
    ids = [IDX_NOSE, IDX_CHIN, IDX_LE, IDX_RE, IDX_LM, IDX_RM]
    return np.array([[lms[i].x * w, lms[i].y * h] for i in ids], dtype=np.float32)

def _bbox_from_landmarks(lms, w, h, pad=0.03):
    xs = [p.x for p in lms]; ys = [p.y for p in lms]
    x1n, y1n = max(0.0, min(xs) - pad), max(0.0, min(ys) - pad)
    x2n, y2n = min(1.0, max(xs) + pad), min(1.0, max(ys) + pad)
    return (int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h))

# Thresholds
YAW_DEG_TRIG, PITCH_UP, PITCH_DOWN = 6, 7, 11
SMOOTH_N, CAPTURE_MIN_MS = 5, 1200

class ProctorDetector:
    def __init__(self):
        self.yaw_hist, self.pitch_hist = deque(maxlen=SMOOTH_N), deque(maxlen=SMOOTH_N)
        self.last_capture_ms, self.noface_streak, self.hand_streak = 0, 0, 0
        self.last_print = 0.0

    def _pose_angles(self, lms, w, h):
        try:
            pts2d = _landmarks_to_pts(lms, w, h)
            cam = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
            ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, cam, np.zeros((4,1)))
            if not ok:
                return None, None
            R, _ = cv2.Rodrigues(rvec)
            _, _, euler = cv2.RQDecomp3x3(R)
            pitch, yaw, _ = map(float, euler)
            return yaw, pitch
        except Exception as e:
            log("POSE_ERR", err=str(e))
            return None, None

    def detect(self, bgr, sid="-", eid="-"):
        try:
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 1)
            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                self.noface_streak += 1
                log("NO_FACE_FRAME", sid, eid, streak=self.noface_streak)
                return "No Face", None, rgb

            self.noface_streak = 0
            lms = res.multi_face_landmarks[0].landmark
            yaw, pitch = self._pose_angles(lms, w, h)
            label = "Looking Forward"

            if yaw is not None and pitch is not None:
                if abs(yaw) > YAW_DEG_TRIG:
                    label = "Looking Left" if yaw < 0 else "Looking Right"
                elif pitch > PITCH_DOWN:
                    label = "Looking Down"
                elif pitch < -PITCH_UP:
                    label = "Looking Up"

            if time.time() - self.last_print > 1.5:
                log("ANGLES", sid, eid, yaw=round(yaw or 0, 2),
                    pitch=round(pitch or 0, 2), label=label)
                self.last_print = time.time()

            log("FACE_DETECTED", sid, eid, label=label)
            return label, _bbox_from_landmarks(lms, w, h), rgb

        except Exception as e:
            log("DETECT_EXCEPTION", sid, eid, err=str(e))
            traceback.print_exc()
            return "Error", None, bgr

    def detect_hands_anywhere(self, rgb, sid="-", eid="-"):
        try:
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                self.hand_streak = 0
                return None
            self.hand_streak += 1
            log("HAND_DETECTED", sid, eid, count=len(res.multi_hand_landmarks))
            return "Hand Detected"
        except Exception as e:
            log("HAND_ERR", sid, eid, err=str(e))
            return None

    def _throttle_ok(self):
        return int(time.time() * 1000) - self.last_capture_ms >= CAPTURE_MIN_MS

    def _mark_captured(self):
        self.last_capture_ms = int(time.time() * 1000)

detectors = defaultdict(ProctorDetector)

# ----------------------------------------------------------------------
# CAPTURE HANDLER — SAVES DIRECTLY TO MYSQL
# ----------------------------------------------------------------------
def _maybe_capture(student_id: str, exam_id: str, bgr, label: str):
    try:
        ok, buf = cv2.imencode(".jpg", bgr)
        if not ok:
            log("CAPTURE_SKIP", student_id, exam_id, reason="encode_failed")
            return

        img_b64 = base64.b64encode(buf).decode("utf-8")
        log("CAPTURE_TRIGGERED", student_id, exam_id, label=label, bytes=len(buf))

        # ✅ Direct database insertion
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert into behavior_logs
        cursor.execute("""
            INSERT INTO behavior_logs (student_id, exam_id, image_base64, warning_type, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (student_id, exam_id, img_b64, label))

        # Update suspicious count
        cursor.execute("""
            UPDATE students SET suspicious_count = suspicious_count + 1 WHERE id = %s
        """, (student_id,))

        conn.commit()
        cursor.close()
        conn.close()

        ts = int(time.time() * 1000)
        last_capture[(student_id, exam_id)] = {"label": label, "at": ts}
        log("LAST_CAPTURE_SET", student_id, exam_id, label=label, at=ts)
    except Exception as e:
        log("CAPTURE_ERR", student_id, exam_id, err=str(e))
        traceback.print_exc()

# ----------------------------------------------------------------------
# WEBRTC OFFER HANDLER
# ----------------------------------------------------------------------
async def _wait_ice_complete(pc):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _(_ev=None):
        if pc.iceGatheringState == "complete":
            done.set()

    await asyncio.wait_for(done.wait(), timeout=5.0)

async def handle_offer(data):
    sid, eid = str(data.get("student_id", "0")), str(data.get("exam_id", "0"))
    log("OFFER_HANDLE", sid, eid)
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _():
        log("CONN_STATE", sid, eid, state=pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)
            for d in (detectors, last_warning, last_metrics, last_capture):
                d.pop((sid, eid), None)
            log("PC_CLOSED", sid, eid)

    @pc.on("track")
    def on_track(track):
        log("TRACK", sid, eid, kind=track.kind)
        if track.kind != "video":
            MediaBlackhole().addTrack(track)
            return

        async def reader():
            det = detectors[(sid, eid)]
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=RECV_TIMEOUT_S)
                    log("FRAME_RECV", sid, eid)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log("TRACK_RECV_ERR", sid, eid, err=str(e))
                    traceback.print_exc()
                    break

                try:
                    bgr = frame.to_ndarray(format="bgr24")
                    head_label, _, rgb = det.detect(bgr, sid, eid)
                    hand_label = det.detect_hands_anywhere(rgb, sid, eid)
                    warn = hand_label or head_label
                    ts = int(time.time() * 1000)
                    last_warning[(sid, eid)] = {"warning": warn, "at": ts}
                    log("DETECTION_RESULT", sid, eid, warn=warn)

                    if det._throttle_ok() and warn not in ("Looking Forward", None, "No Face"):
                        _maybe_capture(sid, eid, bgr, warn)
                        det._mark_captured()
                except Exception as e:
                    log("DETECT_ERR", sid, eid, err=str(e))
                    traceback.print_exc()
                    continue

        asyncio.ensure_future(reader(), loop=_loop)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await _wait_ice_complete(pc)
    return pc.localDescription

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@webrtc_bp.route("/webrtc/offer", methods=["POST"])
def webrtc_offer():
    try:
        data = request.get_json(force=True)
        desc = asyncio.run_coroutine_threadsafe(handle_offer(data), _loop).result()
        return jsonify({"sdp": desc.sdp, "type": desc.type})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@webrtc_bp.route("/webrtc/cleanup", methods=["POST"])
def webrtc_cleanup():
    async def _close_all():
        for pc in list(pcs):
            await pc.close()
            pcs.discard(pc)
    asyncio.run_coroutine_threadsafe(_close_all(), _loop)
    return jsonify({"ok": True})

@webrtc_bp.route("/proctor/last_warning")
def proctor_last_warning():
    sid, eid = request.args.get("student_id"), request.args.get("exam_id")
    if not sid or not eid:
        return jsonify(error="missing student_id or exam_id"), 400
    return jsonify(last_warning.get((sid, eid), {"warning": "Looking Forward", "at": 0}))

@webrtc_bp.route("/proctor/last_capture")
def proctor_last_capture():
    sid, eid = request.args.get("student_id"), request.args.get("exam_id")
    if not sid or not eid:
        return jsonify(error="missing student_id or exam_id"), 400
    return jsonify(last_capture.get((sid, eid), {"label": None, "at": 0}))
