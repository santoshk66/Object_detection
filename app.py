import cv2
import time
import threading
import json
import os
import sys
from queue import Queue
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, Response, request, jsonify, send_file

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

# =====================================================
# GLOBALS
# =====================================================
MODEL_PATH = "yolov8n.pt"
COCO_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
  "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
  "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
  "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
  "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
COCO_IDS = {cls: i for i, cls in enumerate(COCO_CLASSES)}

# Defaults
selected_classes_names = ["person"]
selected_ids = {0}
rtsp_url = "rtsp://192.168.0.8:554/..."
line_ratio = 0.5
record_hourly = True
save_path = "./outputs"
person_in_dir = "left_to_right"
person_out_dir = "right_to_left"
running = False
person_tracks = {}
next_id = 0
count_in = 0
count_out = 0
line_x = None

# Queues and locks
FRAME_QUEUE = Queue(maxsize=2)
RESULT_QUEUE = Queue(maxsize=2)
last_frame = None
frame_lock = threading.Lock()

# Model setup
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    model.to(device)
except ImportError:
    device = 'cpu'
    model = YOLO(MODEL_PATH)

os.makedirs(save_path, exist_ok=True)

# Threads
camera_thread = None
inference_thread = None
process_thread = None
saver_thread = None

IMG_SIZE = 320
DISPLAY_SCALE = 0.6  # Not used for stream
TRACK_TIMEOUT = 2.5
MIN_DIST = 60

print("[✅] YOLO Dashboard Backend initialized!")
print(f"Model: {MODEL_PATH} on {device}")
print(f"Default RTSP: {rtsp_url}")
print(f"Default classes: {selected_classes_names}")
print("--------------------------------------------------")

# =====================================================
# WORKER THREADS
# =====================================================
def camera_reader():
    global rtsp_url, running
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[❌] Failed to open RTSP stream")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        if FRAME_QUEUE.full():
            try:
                FRAME_QUEUE.get_nowait()
            except:
                pass
        FRAME_QUEUE.put(frame)
    cap.release()
    print("[ℹ️] Camera thread stopped")

def inference_worker():
    global running
    while running:
        if FRAME_QUEUE.empty():
            time.sleep(0.01)
            continue
        frame = FRAME_QUEUE.get()
        results = model(frame, imgsz=IMG_SIZE, device=device, verbose=False)
        if RESULT_QUEUE.full():
            try:
                RESULT_QUEUE.get_nowait()
            except:
                pass
        RESULT_QUEUE.put((frame, results))
    print("[ℹ️] Inference thread stopped")

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def process_worker():
    global running, line_x, count_in, count_out, person_tracks, next_id, last_frame
    while running:
        if RESULT_QUEUE.empty():
            time.sleep(0.01)
            continue
        frame, results = RESULT_QUEUE.get()
        h, w = frame.shape[:2]
        if line_x is None:
            line_x = int(w * line_ratio)
        r = results[0]
        if r.boxes is None:
            with frame_lock:
                last_frame = frame.copy()
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        detections = [(get_centroid(b), b, c) for b, c in zip(boxes, classes) if c in selected_ids]
        now = time.time()
        for (cx, cy), box, cls in detections:
            matched_id = None
            min_dist = float('inf')
            for pid, data in person_tracks.items():
                px, py = data['centroid']
                dist = np.hypot(cx - px, cy - py)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = pid
            if matched_id is None or min_dist > MIN_DIST:
                matched_id = next_id
                next_id += 1
                side = "L" if cx < line_x else "R"
                person_tracks[matched_id] = {
                    'centroid': (cx, cy),
                    'side': side,
                    'time': now
                }
            prev_side = person_tracks[matched_id]['side']
            curr_side = "L" if cx < line_x else "R"
            person_tracks[matched_id]['centroid'] = (cx, cy)
            person_tracks[matched_id]['time'] = now
            if prev_side != curr_side:
                if prev_side == "L" and curr_side == "R" and person_in_dir == "left_to_right":
                    count_in += 1
                    person_tracks[matched_id]['side'] = "counted_in"
                elif prev_side == "R" and curr_side == "L" and person_out_dir == "right_to_left":
                    count_out += 1
                    person_tracks[matched_id]['side'] = "counted_out"
                else:
                    person_tracks[matched_id]['side'] = curr_side
            else:
                person_tracks[matched_id]['side'] = curr_side
            # Draw
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{matched_id} {COCO_CLASSES[cls]}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Cleanup old tracks
        to_remove = [pid for pid, d in person_tracks.items() if now - d['time'] > TRACK_TIMEOUT]
        for pid in to_remove:
            del person_tracks[pid]
        # Draw line and counters
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.putText(frame, f"IN: {count_in}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {count_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        with frame_lock:
            last_frame = frame.copy()
    print("[ℹ️] Process thread stopped")

def save_to_excel():
    global count_in, count_out, save_path
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_name = now.strftime("%A")
    hour = now.hour
    next_hour = "00" if hour == 23 else f"{hour + 1:02d}"
    hour_range = f"{hour:02d}:00 - {next_hour}:00"
    data = {
        "Date": date_str,
        "Day": day_name,
        "Hour Range": hour_range,
        "Count IN": count_in,
        "Count OUT": count_out
    }
    df_new = pd.DataFrame([data])
    excel_file = os.path.join(save_path, "people_counter.xlsx")
    if os.path.exists(excel_file):
        df_old = pd.read_excel(excel_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_excel(excel_file, index=False)
    print(f"[💾] Data saved to Excel ({excel_file}) at {hour_range}")
    count_in = 0
    count_out = 0

def saver_worker():
    global record_hourly, running
    while True:
        if record_hourly:
            time.sleep(5)
            if running:
                save_to_excel()
        # else:
        #     time.sleep(10)

# =====================================================
# API ENDPOINTS
# =====================================================
@app.route('/')
def index():
    if os.path.exists('index.html'):
        return send_file('index.html')
    else:
        return "index.html not found. Please place the dashboard HTML as index.html.", 404

@app.route('/stream')
def stream():
    def gen_frames():
        global last_frame
        while True:
            with frame_lock:
                frame = last_frame
            if frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def api_start():
    global running, selected_classes_names, selected_ids, rtsp_url, line_ratio, record_hourly, save_path, line_x, person_tracks, next_id, camera_thread, inference_thread, process_thread
    data = request.get_json()
    if running:
        running = False
        time.sleep(0.5)
    try:
        selected_classes_names = data.get("classes", ["person"])
        selected_ids = set(COCO_IDS[c] for c in selected_classes_names)
    except KeyError as e:
        return jsonify({"error": f"Unknown class: {e.args[0]}"}), 400
    rtsp_url = data.get("rtsp", rtsp_url)
    line_ratio = float(data.get("line_ratio", 0.5))
    record_hourly = data.get("record_hourly", True)
    save_path = data.get("save_path", save_path)
    os.makedirs(save_path, exist_ok=True)
    line_x = None
    person_tracks = {}
    next_id = 0
    running = True
    camera_thread = threading.Thread(target=camera_reader, name="camera", daemon=True)
    camera_thread.start()
    inference_thread = threading.Thread(target=inference_worker, name="inference", daemon=True)
    inference_thread.start()
    process_thread = threading.Thread(target=process_worker, name="process", daemon=True)
    process_thread.start()
    return jsonify({"success": True})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global running
    running = False
    return jsonify({"success": True})

@app.route('/api/counts', methods=['GET'])
def api_counts():
    global count_in, count_out
    return jsonify({"in": count_in, "out": count_out})

@app.route('/api/reset_counts', methods=['POST'])
def api_reset_counts():
    global count_in, count_out
    count_in = 0
    count_out = 0
    return jsonify({"success": True})

@app.route('/api/config', methods=['POST'])
def api_config():
    global person_in_dir, person_out_dir, line_ratio, record_hourly, save_path, rtsp_url
    data = request.get_json()
    person_in_dir = data.get("person_in", person_in_dir)
    person_out_dir = data.get("person_out", person_out_dir)
    line_ratio = float(data.get("line", line_ratio))
    record_hourly = data.get("recorded_save_every_hour", record_hourly)
    save_path = data.get("save_location", save_path)
    rtsp_url = data.get("RTSP", rtsp_url)
    os.makedirs(save_path, exist_ok=True)
    return jsonify({"success": True})

@app.route('/api/status', methods=['GET'])
def api_status():
    global running, selected_classes_names, line_ratio, rtsp_url, record_hourly, save_path, count_in, count_out, person_in_dir, person_out_dir
    return jsonify({
        "running": running,
        "selected_classes": selected_classes_names[:8],  # slice as in JS
        "line_ratio": line_ratio,
        "rtsp": rtsp_url,
        "record_hourly": record_hourly,
        "save_path": save_path,
        "counts": {"in": count_in, "out": count_out},
        "directions": {"in": person_in_dir, "out": person_out_dir}
    })

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    saver_thread = threading.Thread(target=saver_worker, name="saver", daemon=True)
    saver_thread.start()
    print("[🚀] Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)