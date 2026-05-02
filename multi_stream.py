import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from collections import deque

from fn_lanedetection import process_video     #lane detector


try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – YOLO detection disabled.\n"
          "       Run:  pip install ultralytics")


YOLO_MODEL_PATH = "yolov8n.pt"

ROAD_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
    58: "potted plant",
    13: "bench",
    56: "chair",
}


CLASS_CONF_OVERRIDES = {
    0:  0.55,   # person
    56: 0.40,   # chair
    13: 0.40,   # bench
    58: 0.35,   # potted plant
    1:  0.45,   # bicycle
    3:  0.45,   # motorcycle
    2:  0.40,   # car
    5:  0.40,   # bus
    7:  0.40,   # truck
    9:  0.50,   # traffic light
    11: 0.50,   # stop sign
}
CONF_THRESHOLD = 0.25
CLOSE_RATIO     = 0.12
MEDIUM_RATIO    = 0.05
LEFT_ZONE_MAX   = 0.45
RIGHT_ZONE_MIN  = 0.55
CMD_SMOOTH_N    = 5

def get_green_mask(lane_frame):
    if lane_frame is None:
        return None

    h, w = lane_frame.shape[:2]

    hsv = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40], np.uint8)
    upper_green = np.array([85, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask = cv2.dilate(mask, kernel_big, iterations=1)

    return mask

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



def run_yolo(model, frame, roi_mask=None, lane_frame=None):
    green_mask = get_green_mask(lane_frame)

    if model is None:
        return []

    h, w = frame.shape[:2]

    if roi_mask is not None:
        coords = cv2.findNonZero(roi_mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            pad = 10
            rx  = max(0, rx - pad)
            ry  = max(0, ry - pad)
            rw  = min(w - rx, rw + pad * 2)
            rh  = min(h - ry, rh + pad * 2)
            roi_frame = frame[ry:ry+rh, rx:rx+rw]
        else:
            roi_frame = frame
            rx, ry = 0, 0
    else:
        roi_frame = frame
        rx, ry = 0, 0

    results = model(
        roi_frame,
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ROAD_CLASSES:
            continue

        conf = float(box.conf[0])

        # ── Per-class confidence gate
        min_conf = CLASS_CONF_OVERRIDES.get(cls_id, CONF_THRESHOLD)
        if conf < min_conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 += rx;
        x2 += rx
        y1 += ry;
        y2 += ry

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bh_ratio = (y2 - y1) / h

        box_w = x2 - x1
        box_h = y2 - y1
        if box_h == 0:
            continue
        aspect = box_w / box_h

        if cls_id == 0 and aspect > 1.0:
            continue

        #Green mask filter
        if green_mask is not None:
            sample_x = int((x1 + x2) / 2)
            sample_y = min(y2, h - 1)
            patch = green_mask[
                max(0, sample_y - 10): sample_y + 1,
                max(0, sample_x - 15): min(w, sample_x + 15)
            ]
            if patch.size == 0 or patch.max() == 0:
                continue

        detections.append({
            'cls_id': cls_id,
            'label': ROAD_CLASSES[cls_id],
            'conf': conf,
            'box': (x1, y1, x2, y2),
            'cx': cx,
            'cy': cy,
            'bh_ratio': bh_ratio,
        })

    return detections


def proximity_level(bh_ratio):

    if bh_ratio >= CLOSE_RATIO:
        return "CLOSE"
    elif bh_ratio >= MEDIUM_RATIO:
        return "MEDIUM"
    return "FAR"

def compute_command(detections, lane_offset, frame_w, frame_h):
    centre_dets = [d for d in detections
                   if LEFT_ZONE_MAX < d['cx'] < RIGHT_ZONE_MIN]
    left_dets   = [d for d in detections if d['cx'] <= LEFT_ZONE_MAX]
    right_dets  = [d for d in detections if d['cx'] >= RIGHT_ZONE_MIN]

    centre_close = [d for d in centre_dets
                    if proximity_level(d['bh_ratio']) == "CLOSE"]
    if centre_close:
        labels = ", ".join(d['label'] for d in centre_close)
        return "STOP", f"Obstacle CLOSE ahead: {labels}"

    all_close = [d for d in detections
                 if proximity_level(d['bh_ratio']) == "CLOSE"]
    if all_close:
        labels = ", ".join(d['label'] for d in all_close)
        return "STOP", f"Obstacle CLOSE (wide): {labels}"

    signals = [d for d in detections if d['cls_id'] in (9, 11)]
    if signals:
        return "STOP", f"Signal: {signals[0]['label']}"

    blockers = [d for d in centre_dets
                if d['cls_id'] in (58, 13)
                and proximity_level(d['bh_ratio']) in ("CLOSE", "MEDIUM")]
    if blockers:
        labels = ", ".join(d['label'] for d in blockers)
        return "STOP", f"Road blocked by: {labels}"

    centre_person = [d for d in centre_dets
                     if d['cls_id'] == 0
                     and proximity_level(d['bh_ratio']) in ("CLOSE", "MEDIUM")]
    if centre_person:
        return "STOP", "Person in path"

    centre_medium = [d for d in centre_dets
                     if proximity_level(d['bh_ratio']) == "MEDIUM"]
    if centre_medium:
        labels = ", ".join(d['label'] for d in centre_medium)
        return "STOP", f"Obstacle MEDIUM ahead: {labels}"

    if lane_offset is not None:
        if lane_offset < -40:
            return "TURN_RIGHT", f"Lane offset {lane_offset}px"
        elif lane_offset > 40:
            return "TURN_LEFT",  f"Lane offset {lane_offset}px"

    left_near  = [d for d in left_dets
                  if proximity_level(d['bh_ratio']) != "FAR"]
    right_near = [d for d in right_dets
                  if proximity_level(d['bh_ratio']) != "FAR"]

    if left_near and not right_near:
        return "TURN_RIGHT", f"Obstacle left: {left_near[0]['label']}"
    if right_near and not left_near:
        return "TURN_LEFT",  f"Obstacle right: {right_near[0]['label']}"

    return "FORWARD", "Path clear"

class CommandSmoother:
    def __init__(self, n=CMD_SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, cmd):
        self.buf.append(cmd)

    def get(self):
        if not self.buf:
            return "FORWARD"
        return max(set(self.buf), key=self.buf.count)

_CMD_COLORS = {
    "FORWARD"    : (0,   220,  80),
    "STOP"       : (0,    30, 230),
    "TURN_LEFT"  : (0,   200, 255),
    "TURN_RIGHT" : (0,   200, 255),
    "REVERSE"    : (140,   0, 255),
}

_PROX_COLORS = {
    "CLOSE"  : (0,   0,   255),
    "MEDIUM" : (0,  165,  255),
    "FAR"    : (0,  255,    0),
}


def draw_detections(frame, detections):

    out = frame.copy()
    h, w = out.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d['box']
        prox  = proximity_level(d['bh_ratio'])
        color = _PROX_COLORS[prox]

        thickness = 3 if prox == "CLOSE" else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label_txt = f"{d['label']}  {d['conf']:.0%}  [{prox}]"
        (tw, th), _ = cv2.getTextSize(
            label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        by = max(y1 - 4, th + 4)
        cv2.rectangle(out,
                      (x1, by - th - 4), (x1 + tw + 4, by + 2),
                      color, cv2.FILLED)
        cv2.putText(out, label_txt,
                    (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1,
                    cv2.LINE_AA)

        bar_max_h = y2 - y1
        bar_fill  = int(bar_max_h * d['bh_ratio'] / CLOSE_RATIO)
        bar_fill  = min(bar_fill, bar_max_h)
        bx        = x2 + 4
        cv2.rectangle(out,
                      (bx, y2 - bar_fill), (bx + 8, y2),
                      color, cv2.FILLED)
        cv2.rectangle(out,
                      (bx, y1), (bx + 8, y2),
                      (180, 180, 180), 1)

    return out


def draw_command_banner(frame, command, reason, fps_yolo):
    h, w = frame.shape[:2]
    color = _CMD_COLORS.get(command, (200, 200, 200))

    banner_h = 52
    overlay  = frame.copy()
    cv2.rectangle(overlay,
                  (0, h - banner_h), (w, h),
                  color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, command,
                (14, h - banner_h + 36),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.putText(frame, reason,
                (220, h - banner_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1,
                cv2.LINE_AA)

    cv2.putText(frame,
                f"YOLO FPS: {fps_yolo:.1f}",
                (w - 180, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2,
                cv2.LINE_AA)

    return frame


def build_debug_mosaic(lane_frame, raw_edges, raw_hough, yolo_frame):
    h, w = lane_frame.shape[:2]
    th, tw = h // 2, w // 2

    def to_bgr_tile(img, size):
        if img is None:
            return np.zeros((*size[::-1], 3), np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, size)

    t = tw, th

    tl = to_bgr_tile(lane_frame,  t)
    tr = to_bgr_tile(yolo_frame,  t)
    bl = to_bgr_tile(raw_edges,   t)
    br = to_bgr_tile(raw_hough,   t)

    top = np.hstack([tl, tr])
    bot = np.hstack([bl, br])
    return np.vstack([top, bot])


def process_video_week2(video_path: str, show: bool = True):

    model       = _ModelHolder.get()
    cmd_smooth  = CommandSmoother(n=CMD_SMOOTH_N)
    memory = DetectionMemory(front_hold=3, rear_hold=5)
    yolo_times  = deque(maxlen=30)

    frame_skip = 2
    frame_counter = 0

    for pkg in process_video(video_path):
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

        try:
            rear_pkg = next(rear_gen_global)

            rear_frame = rear_pkg["vis"]
            rear_detections = rear_pkg["detections"]
        except StopIteration:
            rear_detections = []

        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        detections = run_yolo(model, frame,
                              roi_mask=roi_edges,
                              lane_frame=lane_frame)  # ← add this
        elapsed = time.perf_counter() - t0
        yolo_times.append(elapsed)
        fps_yolo   = 1.0 / (np.mean(yolo_times) + 1e-9)

        is_ghost = pkg.get("is_ghost", False)
        mode     = pkg.get("mode", "CONFIDENT")

        effective_offset = None if is_ghost else offset

        raw_cmd, reason = compute_command(detections, effective_offset, w, h)
        if raw_cmd == "STOP" and "CLOSE" in reason:

            front_blocked = True

            rear_close = [
                d for d in rear_detections
                if proximity_level(d['bh_ratio']) == "CLOSE"
            ]

            rear_blocked = len(rear_close) > 0

            memory.update_front(front_blocked)
            memory.update_rear(rear_blocked)

            if not memory.rear_present:
                raw_cmd = "REVERSE"
                reason = "Front blocked → Rear clear"
            else:
                raw_cmd = "STOP"
                reason = "Front & Rear blocked"

        else:
            memory.update_front(False)

            rear_close = [
                d for d in rear_detections
                if proximity_level(d['bh_ratio']) == "CLOSE"
            ]

            memory.update_rear(len(rear_close) > 0)
        if is_ghost and raw_cmd == "FORWARD":
            reason = f"[{mode}] {reason}"
        cmd_smooth.update(raw_cmd)
        command = cmd_smooth.get()

        yolo_frame  = draw_detections(lane_frame, detections)
        composite   = draw_command_banner(
            yolo_frame.copy(), command, reason, fps_yolo
        )

        debug_mosaic = build_debug_mosaic(
            lane_frame, edges, raw_hough, yolo_frame
        )

        if show:
            cv2.imshow("Self-Driving | Composite Output", cv2.resize(composite, (640, 360)))
            #cv2.imshow("Debug Mosaic [Lane | YOLO | Edges | Hough]",
                       #debug_mosaic)
            rear_frame = cv2.resize(rear_frame, (640, 360))
            cv2.imshow("Rear Camera", rear_frame)

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
            "fps_lane"    : fps_lane,
            "fps_yolo"    : fps_yolo,
        }

    if show:
        cv2.destroyAllWindows()



class MultiStreamProcessor:
    def __init__(self, video_paths: list, show: bool = False):
        self.paths  = video_paths
        self.show   = show
        self._q     = queue.Queue(maxsize=64)
        self._stop  = threading.Event()
        self._threads = []

    def _worker(self, path):
        try:
            for pkg in process_video_week2(path, show=False):
                if self._stop.is_set():
                    break
                pkg["source"] = Path(path).name
                self._q.put(pkg)
        except Exception as exc:
            print(f"[ERROR] Stream {path}: {exc}")
        finally:
            self._q.put(None)

    def start(self):
        for p in self.paths:
            t = threading.Thread(target=self._worker, args=(p,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def results(self):
        finished = 0
        total    = len(self.paths)
        while finished < total:
            pkg = self._q.get()
            if pkg is None:
                finished += 1
                continue
            yield pkg


def log_frame(pkg, frame_idx: int):
    src    = pkg.get("source", "video")
    cmd    = pkg["command"]
    reason = pkg["reason"]
    n_det  = len(pkg["detections"])
    off    = pkg["lane_offset"]
    fl     = pkg["fps_lane"]
    fy     = pkg["fps_yolo"]

    off_str = f"{off:+d}px" if off is not None else "N/A"
    print(
        f"[{src}] frame={frame_idx:05d}  CMD={cmd:<11s}  "
        f"offset={off_str:<8s}  objs={n_det}  "
        f"lane_fps={fl:.1f}  yolo_fps={fy:.1f}  | {reason}"
    )
def process_rear_only(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 2
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

        yield {
            "frame": frame,
            "vis": vis,
            "detections": detections
        }

REAR_VIDEO_PATH = r"C:\Users\rzrid\Desktop\DIP\Project\MultiStreamDataset\Video1\Back.mp4"
rear_gen_global = None

class DetectionMemory:
    def __init__(self, front_hold=3, rear_hold=5):
        self.front_hold = front_hold
        self.rear_hold = rear_hold

        self.front_counter = 0
        self.rear_counter = 0

        self.front_present = False
        self.rear_present = False

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
def main():
    global rear_gen_global
    model = _ModelHolder.get()
    rear_gen_global = process_rear_only(REAR_VIDEO_PATH, model)
    videos = [
        r"C:\Users\rzrid\Desktop\DIP\Project\MultiStreamDataset\Video1\Front.mp4",
    ]

    for vid in videos:
        print(f"\n{'='*60}\nProcessing (single-stream): {vid}\n{'='*60}")
        for idx, pkg in enumerate(process_video_week2(vid, show=True)):
            log_frame(pkg, idx)


    print("\nAll videos processed.")


if __name__ == "__main__":
    main()