import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from collections import deque


from fn_lanedetection import process_video          # lane detector


try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – YOLO disabled.\n"
          "       pip install ultralytics")

# constants
YOLO_MODEL_PATH = "yolov8n.pt"

YOLO_STRIDE     = 2
DISPLAY_STRIDE  = 2
LOG_STRIDE      = 10
KMEANS_STRIDE   = 4

CONF_THRESHOLD  = 0.25
CLOSE_RATIO     = 0.12
MEDIUM_RATIO    = 0.05
LEFT_ZONE_MAX   = 0.45
RIGHT_ZONE_MIN  = 0.55
CMD_SMOOTH_N    = 5

ROAD_CLASSES = {
    0:  "person",      1:  "bicycle",     2:  "car",
    3:  "motorcycle",  5:  "bus",         7:  "truck",
    9:  "traffic light", 11: "stop sign",
    58: "potted plant", 13: "bench",      56: "chair",
}

CLASS_CONF_OVERRIDES = {
    0: 0.55, 56: 0.40, 13: 0.40, 58: 0.35,
    1: 0.45, 3:  0.45, 2:  0.40, 5:  0.40,
    7: 0.40, 9:  0.50, 11: 0.50,
}

_CMD_COLORS = {
    "FORWARD"    : (0,  220,  80),
    "STOP"       : (0,   30, 230),
    "TURN_LEFT"  : (0,  200, 255),
    "TURN_RIGHT" : (0,  200, 255),
    "REVERSE"    : (140,  0, 255),
}
_PROX_COLORS = {
    "CLOSE" : (0,   0, 255),
    "MEDIUM": (0, 165, 255),
    "FAR"   : (0, 255,   0),
}


# YOLO model singleton
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


# Green-mask cache
class _GreenMaskCache:
    def __init__(self):
        self._last_id  = None
        self._last_mask = None

    def get(self, lane_frame):
        if lane_frame is None:
            return None
        fid = id(lane_frame)
        if fid == self._last_id:
            return self._last_mask
        self._last_id   = fid
        self._last_mask = _compute_green_mask(lane_frame)
        return self._last_mask


def _compute_green_mask(lane_frame):
    hsv = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([35,  40,  40], np.uint8),
                       np.array([85, 255, 255], np.uint8))
    k7 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    mask = cv2.dilate(mask,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40)),
                      iterations=1)
    return mask



# Threaded YOLO worker
class YOLOWorker:

    def __init__(self, model, mask_cache: _GreenMaskCache):
        self._model      = model
        self._mask_cache = mask_cache
        self._in_q       = queue.Queue(maxsize=2)   # bounded → never back-logs
        self._out_q      = queue.Queue(maxsize=8)
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, lane_frame, roi_edges):
        try:
            self._in_q.put_nowait((frame, lane_frame, roi_edges))
        except queue.Full:
            pass

    def latest(self):
        result = None
        while True:
            try:
                result = self._out_q.get_nowait()
            except queue.Empty:
                break
        return result

    def _run(self):
        while True:
            frame, lane_frame, roi_edges = self._in_q.get()
            try:
                dets = _run_yolo_sync(self._model, frame,
                                      roi_edges, lane_frame,
                                      self._mask_cache)
                self._out_q.put(dets)
            except Exception as exc:
                print(f"[YOLO-thread ERROR] {exc}")
                self._out_q.put([])


def _run_yolo_sync(model, frame, roi_mask, lane_frame, mask_cache):
    if model is None:
        return []

    green_mask = mask_cache.get(lane_frame)
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

    results = model(roi_frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)[0]

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

        bh_ratio = bh_px / h
        cx       = ((x1 + x2) / 2) / w

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
            'cy'      : ((y1 + y2) / 2) / h,
            'bh_ratio': bh_ratio,
        })
    return detections


# Command logic
def proximity_level(bh_ratio):
    if bh_ratio >= CLOSE_RATIO:  return "CLOSE"
    if bh_ratio >= MEDIUM_RATIO: return "MEDIUM"
    return "FAR"


def compute_command(detections, lane_offset, frame_w, frame_h):
    centre = [d for d in detections if LEFT_ZONE_MAX < d['cx'] < RIGHT_ZONE_MIN]
    left   = [d for d in detections if d['cx'] <= LEFT_ZONE_MAX]
    right  = [d for d in detections if d['cx'] >= RIGHT_ZONE_MIN]

    close_centre = [d for d in centre if proximity_level(d['bh_ratio']) == "CLOSE"]
    if close_centre:
        return "STOP", "Obstacle CLOSE ahead: " + ", ".join(d['label'] for d in close_centre)

    all_close = [d for d in detections if proximity_level(d['bh_ratio']) == "CLOSE"]
    if all_close:
        return "STOP", "Obstacle CLOSE (wide): " + ", ".join(d['label'] for d in all_close)

    signals = [d for d in detections if d['cls_id'] in (9, 11)]
    if signals:
        return "STOP", f"Signal: {signals[0]['label']}"

    blockers = [d for d in centre if d['cls_id'] in (58, 13)
                and proximity_level(d['bh_ratio']) in ("CLOSE", "MEDIUM")]
    if blockers:
        return "STOP", "Road blocked by: " + ", ".join(d['label'] for d in blockers)

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

    left_near  = [d for d in left  if proximity_level(d['bh_ratio']) != "FAR"]
    right_near = [d for d in right if proximity_level(d['bh_ratio']) != "FAR"]
    if left_near  and not right_near: return "TURN_RIGHT", f"Obstacle left: {left_near[0]['label']}"
    if right_near and not left_near:  return "TURN_LEFT",  f"Obstacle right: {right_near[0]['label']}"

    return "FORWARD", "Path clear"


class CommandSmoother:
    def __init__(self, n=CMD_SMOOTH_N):
        self.buf = deque(maxlen=n)
    def update(self, cmd):  self.buf.append(cmd)
    def get(self):
        if not self.buf: return "FORWARD"
        return max(set(self.buf), key=self.buf.count)


# Drawing helpers
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
        bar_max = y2 - y1
        bar_fill = min(int(bar_max * d['bh_ratio'] / CLOSE_RATIO), bar_max)
        bx = x2 + 4
        cv2.rectangle(out, (bx, y2-bar_fill), (bx+8, y2), color, cv2.FILLED)
        cv2.rectangle(out, (bx, y1),          (bx+8, y2), (180,180,180), 1)
    return out


def draw_command_banner(frame, command, reason, fps_yolo, fps_total):
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
    cv2.putText(frame, f"FPS: {fps_total:.1f} | YOLO: {fps_yolo:.1f}", (w-280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2, cv2.LINE_AA)
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

# FPS Counter class
class FPSCounter:
    def __init__(self, window_size=30):
        self._window = deque(maxlen=window_size)
        self._start_time = time.perf_counter()
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


# Main  pipeline
def process_video_week2(video_path: str, show: bool = True, fps_counter: FPSCounter = None):

    if fps_counter is None:
        fps_counter = FPSCounter()

    model       = _ModelHolder.get()
    mask_cache  = _GreenMaskCache()
    yolo_worker = YOLOWorker(model, mask_cache) if model else None

    cmd_smooth  = CommandSmoother(n=CMD_SMOOTH_N)
    yolo_times  = deque(maxlen=30)

    last_detections = []
    frame_idx       = 0
    t_yolo_submit   = time.perf_counter()

    for pkg in process_video(video_path):
        # Update FPS counter
        fps_counter.update()

        frame      = pkg["frame"]
        lane_frame = pkg["lane_frame"]
        edges      = pkg["edges"]
        roi_edges  = pkg["roi_edges"]
        raw_hough  = pkg["raw_debug"]
        offset     = pkg["offset"]
        fps_lane   = pkg["fps"]
        is_ghost   = pkg.get("is_ghost", False)
        mode       = pkg.get("mode", "CONFIDENT")

        h, w = frame.shape[:2]

        if yolo_worker is not None and frame_idx % YOLO_STRIDE == 0:
            t_yolo_submit = time.perf_counter()
            yolo_worker.submit(frame, lane_frame, roi_edges)

        if yolo_worker is not None:
            fresh = yolo_worker.latest()
            if fresh is not None:
                elapsed = time.perf_counter() - t_yolo_submit
                yolo_times.append(elapsed)
                last_detections = fresh

        detections = last_detections
        fps_yolo   = 1.0 / (np.mean(yolo_times) + 1e-9) if yolo_times else 0.0
        fps_total  = fps_counter.current_fps

        effective_offset = None if is_ghost else offset
        raw_cmd, reason  = compute_command(detections, effective_offset, w, h)
        if is_ghost and raw_cmd == "FORWARD":
            reason = f"[{mode}] {reason}"
        cmd_smooth.update(raw_cmd)
        command = cmd_smooth.get()

        yolo_frame   = draw_detections(lane_frame, detections)
        composite    = draw_command_banner(yolo_frame.copy(), command, reason, fps_yolo, fps_total)
        debug_mosaic = build_debug_mosaic(lane_frame, edges, raw_hough, yolo_frame)

        # display
        if show and frame_idx % DISPLAY_STRIDE == 0:
            cv2.imshow("Self-Driving | Composite Output", composite)
            cv2.imshow("Debug Mosaic [Lane | YOLO | Edges | Hough]", debug_mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        frame_idx += 1

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
            "fps_total"   : fps_total,
            "is_ghost"    : is_ghost,
            "roi_edges"   : roi_edges,
            "frame_idx"   : frame_idx,
        }

    if show:
        cv2.destroyAllWindows()


def log_frame(pkg, frame_idx: int):
    if frame_idx % LOG_STRIDE != 0:
        return
    src     = pkg.get("source", "video")
    off     = pkg["lane_offset"]
    off_str = f"{off:+d}px" if off is not None else "N/A"
    print(
        f"[{src}] frame={frame_idx:05d}  CMD={pkg['command']:<11s}  "
        f"offset={off_str:<8s}  objs={len(pkg['detections'])}  "
        f"FPS={pkg['fps_total']:.1f}  lane_fps={pkg['fps_lane']:.1f}  yolo_fps={pkg['fps_yolo']:.1f}"
        f"  | {pkg['reason']}"
    )


def main():

    # videos provided ...
    videos = [
          r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_043754655.TS.mp4",
         r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_043922504.TS.mp4",
          r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044505516.TS.mp4",
          r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044603023.TS.mp4",
         r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044746327.TS.mp4",
          r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_045117252.TS.mp4",
     ]


    # multistreaming videos
    videos = [
       r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\MultiStreamDataset\Video1\Front.mp4",
       r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\MultiStreamDataset\Video2\Front.mp4",
       r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\MultiStreamDataset\Video3\Front.mp4",
       r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\MultiStreamDataset\Video4\Front.mp4",
    ]

    #my dataset Day
    videos = [
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\simple_mydataset\vid1.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\simple_mydataset\vid2.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\simple_mydataset\vid3.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\simple_mydataset\vid4.mp4",
    ]

    #my dataset Night
    videos = [
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\night_mydataset\vid1.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\night_mydataset\vid2.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\night_mydataset\vid3.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\night_mydataset\vid4.mp4",

    ]


    for vid_idx, vid in enumerate(videos):
        print(f"\n{'='*60}")
        print(f"Processing video {vid_idx+1}/{len(videos)}: {Path(vid).name}")
        print(f"{'='*60}")

        fps_counter = FPSCounter()

        for idx, pkg in enumerate(process_video_week2(vid, show=True, fps_counter=fps_counter)):
            log_frame(pkg, idx)

        # Print FPS summary for this video
        print(f"\n{'─'*60}")
        print(f"FPS Summary for {Path(vid).name}:")
        print(f"  Total frames: {fps_counter.total_frames}")
        print(f"  Elapsed time: {fps_counter.elapsed_time:.2f} seconds")
        print(f"  Average FPS:  {fps_counter.average_fps:.2f}")
        print(f"{'─'*60}\n")

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()