import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from collections import deque

from fn_lanedetection import process_video

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – YOLO detection disabled.\n"
          "       Run:  pip install ultralytics")

YOLO_MODEL_PATH = "yolov8n.pt"

ROAD_CLASSES = {
    0:  "person",       1:  "bicycle",      2:  "car",
    3:  "motorcycle",   5:  "bus",           7:  "truck",
    9:  "traffic light", 11: "stop sign",
    58: "potted plant", 13: "bench",        56: "chair",
}

CLASS_CONF_OVERRIDES = {
    0:  0.55, 56: 0.40, 13: 0.40, 58: 0.35,
    1:  0.45,  3: 0.45,  2: 0.40,  5: 0.40,
    7:  0.40,  9: 0.50, 11: 0.50,
}

CONF_THRESHOLD = 0.25
CLOSE_RATIO    = 0.12
MEDIUM_RATIO   = 0.05
LEFT_ZONE_MAX  = 0.45
RIGHT_ZONE_MIN = 0.55
CMD_SMOOTH_N   = 5

_CMD_COLORS = {
    "FORWARD"    : (0,   220,  80),
    "STOP"       : (0,    30, 230),
    "TURN_LEFT"  : (0,   200, 255),
    "TURN_RIGHT" : (0,   200, 255),
    "REVERSE"    : (140,   0, 255),
}
_PROX_COLORS = {
    "CLOSE"  : (0,   0, 255),
    "MEDIUM" : (0, 165, 255),
    "FAR"    : (0, 255,   0),
}


# ── Model singleton ────────────────────────────────────────────────────────────
class _ModelHolder:
    _model = None
    _lock  = threading.Lock()

    @classmethod
    def get(cls):
        if not _YOLO_AVAILABLE:
            return None
        with cls._lock:
            if cls._model is None:
                print(f"[INFO] Loading YOLO model: {YOLO_MODEL_PATH} …")
                cls._model = _YOLO(YOLO_MODEL_PATH)
                cls._model.fuse()
                print("[INFO] YOLO model ready.")
        return cls._model


# ── Green mask ────────────────────────────────────────────────────────────────
def get_green_mask(lane_frame):
    if lane_frame is None:
        return None
    hsv = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([35, 40,  40], np.uint8),
                       np.array([85, 255, 255], np.uint8))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask = cv2.dilate(mask, kernel_big, iterations=1)
    return mask


# ── YOLO inference ────────────────────────────────────────────────────────────
def run_yolo(model, frame, roi_mask=None, lane_frame=None):
    green_mask = get_green_mask(lane_frame)
    if model is None:
        return []

    h, w = frame.shape[:2]
    rx, ry = 0, 0

    if roi_mask is not None:
        coords = cv2.findNonZero(roi_mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            pad = 10
            rx  = max(0, rx - pad);   ry  = max(0, ry - pad)
            rw  = min(w - rx, rw + pad * 2)
            rh  = min(h - ry, rh + pad * 2)
            roi_frame = frame[ry:ry+rh, rx:rx+rw]
        else:
            roi_frame = frame
    else:
        roi_frame = frame

    results = model(roi_frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ROAD_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < CLASS_CONF_OVERRIDES.get(cls_id, CONF_THRESHOLD):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 += rx; x2 += rx; y1 += ry; y2 += ry
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))

        bw = x2 - x1; bh_px = y2 - y1
        if bh_px == 0:
            continue
        if cls_id == 0 and (bw / bh_px) > 1.0:
            continue

        cx       = ((x1 + x2) / 2) / w
        cy       = ((y1 + y2) / 2) / h
        bh_ratio = bh_px / h

        if green_mask is not None:
            sx = int((x1 + x2) / 2)
            sy = min(y2, h - 1)
            patch = green_mask[max(0, sy-10):sy+1,
                                max(0, sx-15):min(w, sx+15)]
            if patch.size == 0 or patch.max() == 0:
                continue

        detections.append({
            'cls_id'  : cls_id,
            'label'   : ROAD_CLASSES[cls_id],
            'conf'    : conf,
            'box'     : (x1, y1, x2, y2),
            'cx'      : cx,
            'cy'      : cy,
            'bh_ratio': bh_ratio,
        })
    return detections


# ── Proximity ──────────────────────────────────────────────────────────────────
def proximity_level(bh_ratio):
    if bh_ratio >= CLOSE_RATIO:  return "CLOSE"
    if bh_ratio >= MEDIUM_RATIO: return "MEDIUM"
    return "FAR"


# ── Command logic ─────────────────────────────────────────────────────────────
def compute_command(detections, lane_offset, frame_w, frame_h):
    centre = [d for d in detections if LEFT_ZONE_MAX < d['cx'] < RIGHT_ZONE_MIN]
    left   = [d for d in detections if d['cx'] <= LEFT_ZONE_MAX]
    right  = [d for d in detections if d['cx'] >= RIGHT_ZONE_MIN]

    cc = [d for d in centre if proximity_level(d['bh_ratio']) == "CLOSE"]
    if cc:
        return "STOP", "Obstacle CLOSE ahead: " + ", ".join(d['label'] for d in cc)

    ac = [d for d in detections if proximity_level(d['bh_ratio']) == "CLOSE"]
    if ac:
        return "STOP", "Obstacle CLOSE (wide): " + ", ".join(d['label'] for d in ac)

    sigs = [d for d in detections if d['cls_id'] in (9, 11)]
    if sigs:
        return "STOP", f"Signal: {sigs[0]['label']}"

    blk = [d for d in centre if d['cls_id'] in (58, 13)
           and proximity_level(d['bh_ratio']) in ("CLOSE", "MEDIUM")]
    if blk:
        return "STOP", "Road blocked by: " + ", ".join(d['label'] for d in blk)

    cp = [d for d in centre if d['cls_id'] == 0
          and proximity_level(d['bh_ratio']) in ("CLOSE", "MEDIUM")]
    if cp:
        return "STOP", "Person in path"

    cm = [d for d in centre if proximity_level(d['bh_ratio']) == "MEDIUM"]
    if cm:
        return "STOP", "Obstacle MEDIUM ahead: " + ", ".join(d['label'] for d in cm)

    if lane_offset is not None:
        if lane_offset < -40: return "TURN_RIGHT", f"Lane offset {lane_offset}px"
        if lane_offset >  40: return "TURN_LEFT",  f"Lane offset {lane_offset}px"

    ln = [d for d in left  if proximity_level(d['bh_ratio']) != "FAR"]
    rn = [d for d in right if proximity_level(d['bh_ratio']) != "FAR"]
    if ln and not rn: return "TURN_RIGHT", f"Obstacle left: {ln[0]['label']}"
    if rn and not ln: return "TURN_LEFT",  f"Obstacle right: {rn[0]['label']}"

    return "FORWARD", "Path clear"


# ── Command smoother ──────────────────────────────────────────────────────────
class CommandSmoother:
    def __init__(self, n=CMD_SMOOTH_N):
        self.buf = deque(maxlen=n)
    def update(self, cmd): self.buf.append(cmd)
    def get(self):
        if not self.buf: return "FORWARD"
        return max(set(self.buf), key=self.buf.count)


# ── Detection memory ──────────────────────────────────────────────────────────
class DetectionMemory:
    def __init__(self, front_hold=3, rear_hold=5):
        self.front_hold    = front_hold
        self.rear_hold     = rear_hold
        self.front_counter = 0
        self.rear_counter  = 0
        self.front_present = False
        self.rear_present  = False

    def update_front(self, detected):
        if detected:
            self.front_counter = self.front_hold
            self.front_present = True
        else:
            self.front_counter -= 1
            if self.front_counter <= 0:
                self.front_present = False

    def update_rear(self, detected):
        if detected:
            self.rear_counter = self.rear_hold
            self.rear_present = True
        else:
            self.rear_counter -= 1
            if self.rear_counter <= 0:
                self.rear_present = False


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    out = frame.copy()
    h, w = out.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d['box']
        prox  = proximity_level(d['bh_ratio'])
        color = _PROX_COLORS[prox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if prox == "CLOSE" else 2)
        lbl = f"{d['label']}  {d['conf']:.0%}  [{prox}]"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        by = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, by-th-4), (x1+tw+4, by+2), color, cv2.FILLED)
        cv2.putText(out, lbl, (x1+2, by-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1, cv2.LINE_AA)
        bar_max  = y2 - y1
        bar_fill = min(int(bar_max * d['bh_ratio'] / CLOSE_RATIO), bar_max)
        bx = x2 + 4
        cv2.rectangle(out, (bx, y2-bar_fill), (bx+8, y2), color, cv2.FILLED)
        cv2.rectangle(out, (bx, y1),          (bx+8, y2), (180,180,180), 1)
    return out


def draw_command_banner(frame, command, reason, fps_yolo):
    h, w  = frame.shape[:2]
    color = _CMD_COLORS.get(command, (200, 200, 200))
    bh    = 52
    ov    = frame.copy()
    cv2.rectangle(ov, (0, h-bh), (w, h), color, cv2.FILLED)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, command, (14, h-bh+36),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, reason, (220, h-bh+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240,240,240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"YOLO FPS: {fps_yolo:.1f}", (w-180, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,200,0), 2, cv2.LINE_AA)
    return frame


def build_debug_mosaic(lane_frame, raw_edges, raw_hough, yolo_frame):
    h, w   = lane_frame.shape[:2]
    th, tw = h // 2, w // 2
    t      = (tw, th)

    def tile(img):
        if img is None:
            return np.zeros((th, tw, 3), np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, t)

    return np.vstack([
        np.hstack([tile(lane_frame), tile(yolo_frame)]),
        np.hstack([tile(raw_edges),  tile(raw_hough)]),
    ])


# ── Rear-only generator ───────────────────────────────────────────────────────
def process_rear_only(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_skip    = 2
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue
        detections = run_yolo(model, frame)
        vis = draw_detections(frame, detections)
        yield {"frame": frame, "vis": vis, "detections": detections}
    cap.release()


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window_size=30):
        self._window      = deque(maxlen=window_size)
        self._start_time  = time.perf_counter()
        self._total_frames = 0

    def update(self):
        now = time.perf_counter()
        self._window.append(now)
        self._total_frames += 1

    @property
    def current_fps(self):
        if len(self._window) < 2:
            return 0.0
        dt = self._window[-1] - self._window[0]
        return (len(self._window) - 1) / dt if dt > 0 else 0.0

    @property
    def average_fps(self):
        elapsed = time.perf_counter() - self._start_time
        return self._total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def elapsed_time(self):
        return time.perf_counter() - self._start_time


# ── Main dual-camera pipeline ─────────────────────────────────────────────────
def process_video_dual(front_path: str, rear_path: str,
                        model, show: bool = True,
                        fps_counter: FPSCounter = None):
    """
    Processes one front+rear video pair.
    Yields per-frame packages with all fields needed by the metrics collector.
    """
    if fps_counter is None:
        fps_counter = FPSCounter()

    rear_gen   = process_rear_only(rear_path, model)
    cmd_smooth = CommandSmoother(n=CMD_SMOOTH_N)
    memory     = DetectionMemory(front_hold=3, rear_hold=5)
    yolo_times = deque(maxlen=30)

    frame_skip    = 2
    frame_counter = 0

    for pkg in process_video(front_path):
        fps_counter.update()
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame      = pkg["frame"]
        lane_frame = pkg["lane_frame"]
        edges      = pkg["edges"]
        roi_edges  = pkg["roi_edges"]
        raw_hough  = pkg["raw_debug"]
        offset     = pkg["offset"]
        fps_lane   = pkg["fps"]
        is_ghost   = pkg.get("is_ghost", False)
        mode       = pkg.get("mode", "CONFIDENT")

        # Consume one rear frame
        try:
            rear_pkg        = next(rear_gen)
            rear_frame_vis  = rear_pkg["vis"]
            rear_detections = rear_pkg["detections"]
        except StopIteration:
            rear_detections = []
            rear_frame_vis  = None

        h, w = frame.shape[:2]

        t0         = time.perf_counter()
        detections = run_yolo(model, frame, roi_mask=roi_edges, lane_frame=lane_frame)
        elapsed    = time.perf_counter() - t0
        yolo_times.append(elapsed)
        fps_yolo = 1.0 / (np.mean(yolo_times) + 1e-9)

        effective_offset = None if is_ghost else offset
        raw_cmd, reason  = compute_command(detections, effective_offset, w, h)

        # Dual-camera REVERSE logic
        if raw_cmd == "STOP" and "CLOSE" in reason:
            rear_close   = [d for d in rear_detections
                            if proximity_level(d['bh_ratio']) == "CLOSE"]
            rear_blocked = len(rear_close) > 0
            memory.update_front(True)
            memory.update_rear(rear_blocked)
            if not memory.rear_present:
                raw_cmd = "REVERSE"
                reason  = "Front blocked → Rear clear"
            else:
                raw_cmd = "STOP"
                reason  = "Front & Rear blocked"
        else:
            memory.update_front(False)
            rear_close = [d for d in rear_detections
                          if proximity_level(d['bh_ratio']) == "CLOSE"]
            memory.update_rear(len(rear_close) > 0)

        if is_ghost and raw_cmd == "FORWARD":
            reason = f"[{mode}] {reason}"

        cmd_smooth.update(raw_cmd)
        command = cmd_smooth.get()

        yolo_frame   = draw_detections(lane_frame, detections)
        composite    = draw_command_banner(yolo_frame.copy(), command, reason, fps_yolo)
        debug_mosaic = build_debug_mosaic(lane_frame, edges, raw_hough, yolo_frame)

        if show:
            cv2.imshow("Front | Composite Output", cv2.resize(composite, (640, 360)))
            if rear_frame_vis is not None:
                cv2.imshow("Rear Camera", cv2.resize(rear_frame_vis, (640, 360)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        yield {
            "frame"       : frame,
            "lane_frame"  : lane_frame,
            "yolo_frame"  : yolo_frame,
            "composite"   : composite,
            "debug_mosaic": debug_mosaic,
            "detections"  : detections,
            "command"     : command,
            "reason"      : reason,
            "lane_offset" : offset,
            "offset"      : offset,          # needed by metrics collector
            "fps_lane"    : fps_lane,
            "fps_yolo"    : fps_yolo,
            "fps_total"   : fps_counter.current_fps,
            "is_ghost"    : is_ghost,
            "roi_edges"   : roi_edges,
        }

    if show:
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from dual_camera_metrics import DualCameraMetricsCollector

    BASE_DIR = r"C:\Users\rzrid\Desktop\DIP\Project\MultiStreamDataset"

    # All four video pairs — naming: VideoN/Front.mp4 and VideoN/Back.mp4
    video_pairs = [
        (rf"{BASE_DIR}\Video1\Front.mp4", rf"{BASE_DIR}\Video1\Back.mp4"),
        (rf"{BASE_DIR}\Video2\Front.mp4", rf"{BASE_DIR}\Video2\Back.mp4"),
        (rf"{BASE_DIR}\Video3\Front.mp4", rf"{BASE_DIR}\Video3\Back.mp4"),
        (rf"{BASE_DIR}\Video4\Front.mp4", rf"{BASE_DIR}\Video4\Back.mp4"),
    ]

    model = _ModelHolder.get()

    for front_path, rear_path in video_pairs:
        pair_name = Path(front_path).parent.name   # "Video1", "Video2", etc.

        print(f"\n{'='*60}")
        print(f"  Processing pair: {pair_name}")
        print(f"  Front : {front_path}")
        print(f"  Rear  : {rear_path}")
        print(f"{'='*60}")

        fps_counter = FPSCounter()
        collector   = DualCameraMetricsCollector(pair_name)

        for idx, pkg in enumerate(process_video_dual(
                front_path, rear_path, model,
                show=True, fps_counter=fps_counter)):
            collector.record(pkg, idx)

        collector.print_summary(fps_counter)

    print("\nAll video pairs processed.")


if __name__ == "__main__":
    main()
