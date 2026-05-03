
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


#Helpers
CLOSE_RATIO  = 0.12
MEDIUM_RATIO = 0.05


def _edge_density(roi_edges) -> float:
    if roi_edges is None or roi_edges.size == 0:
        return 0.0
    return float(np.count_nonzero(roi_edges)) / roi_edges.size


def _lane_confidence(pkg) -> float:
    offset = pkg.get("offset")
    both   = offset is not None and not pkg.get("is_ghost", False)
    one    = pkg.get("is_ghost", False) and offset is not None
    lane_score = 1.0 if both else (0.5 if one else 0.0)
    ed         = _edge_density(pkg.get("roi_edges"))
    edge_score = min(ed / 0.12, 1.0)
    return round(lane_score * 55.0 + edge_score * 45.0, 2)


def _prox(d: dict) -> str:
    bh = d.get("bh_ratio", 0)
    if bh >= CLOSE_RATIO:  return "CLOSE"
    if bh >= MEDIUM_RATIO: return "MEDIUM"
    return "FAR"


# ── Frame record ──────────────────────────────────────────────────────────────
@dataclass
class DualFrameRecord:
    frame_id:         int
    timestamp:        float
    lane_fps:         float
    yolo_fps:         float
    is_ghost:         bool
    lane_confidence:  float
    both_lanes:       bool
    lane_offset:      Optional[int]
    edge_density:     float
    n_detections:     int
    command:          str
    det_classes:      list
    det_proximities:  list


# ── Collector
class DualCameraMetricsCollector:

    SLOW_LANE_FPS = 12.0
    SLOW_YOLO_FPS = 12.0

    def __init__(self, pair_name: str):
        self.pair_name   = pair_name
        self._frames     = []
        self._start_time = None
        self._end_time   = None

    def record(self, pkg: dict, frame_id: int):
        detections = pkg.get("detections", [])
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
        self._end_time = now

        self._frames.append(DualFrameRecord(
            frame_id        = frame_id,
            timestamp       = now,
            lane_fps        = float(pkg.get("fps_lane", 0) or 0),
            yolo_fps        = float(pkg.get("fps_yolo", 0) or 0),
            is_ghost        = bool(pkg.get("is_ghost", False)),
            lane_confidence = _lane_confidence(pkg),
            both_lanes      = (pkg.get("offset") is not None
                               and not pkg.get("is_ghost", False)),
            lane_offset     = pkg.get("lane_offset") or pkg.get("offset"),
            edge_density    = _edge_density(pkg.get("roi_edges")),
            n_detections    = len(detections),
            command         = pkg.get("command", "FORWARD"),
            det_classes     = [d["label"] for d in detections],
            det_proximities = [_prox(d)   for d in detections],
        ))

    def print_summary(self, fps_counter=None):
        _print_dual_report(self.pair_name, self._frames,
                           self._start_time, self._end_time,
                           fps_counter)


# ── Report printer ────
def _print_dual_report(pair_name, frames, start_time, end_time, fps_counter=None):
    SEP  = "=" * 68
    SEP2 = "-" * 68
    n    = len(frames)

    if n == 0:
        print(f"\n[Metrics] No frames recorded for {pair_name}")
        return

    print(f"\n{SEP}")
    print(f"  DUAL-CAMERA METRICS  —  {pair_name}")
    print(SEP)

    # ── 1. Real-time performance
    print(f"\n  ┌─ 1. REAL-TIME PERFORMANCE {'─'*39}")

    if start_time is not None and end_time is not None:
        total_time = max(end_time - start_time, 1e-6)
        true_fps   = n / total_time
        print(f"  │  {'True pipeline FPS':<22}  {true_fps:6.2f} fps  (wall-clock based)")
        print(f"  │  {'Total processing time':<22}  {total_time:.2f} sec")
    else:
        print(f"  │  True pipeline FPS  :  (no timing data)")

    if fps_counter is not None:
        print(f"  │  {'Average FPS (counter)':<22}  {fps_counter.average_fps:.2f} fps")
        print(f"  │  {'Total frames (counter)':<22}  {fps_counter.total_frames}")

    lane_fps_vals = [f.lane_fps for f in frames if f.lane_fps > 0]
    yolo_fps_vals = [f.yolo_fps for f in frames if f.yolo_fps > 0]

    def fps_row(label, vals, thresh):
        if not vals:
            print(f"  │  {label:<22}  no data")
            return
        avg  = np.mean(vals)
        mn   = np.min(vals)
        mx   = np.max(vals)
        slow = sum(1 for v in vals if v < thresh) / len(vals) * 100
        print(f"  │  {label:<22}  avg={avg:5.1f}  min={mn:5.1f}  max={mx:5.1f}  slow(<{thresh:.0f}fps)={slow:4.1f}%")

    fps_row("Lane detection FPS",  lane_fps_vals, DualCameraMetricsCollector.SLOW_LANE_FPS)
    fps_row("YOLO detection FPS",  yolo_fps_vals, DualCameraMetricsCollector.SLOW_YOLO_FPS)
    print(f"  │  {'Total frames processed':<22}  {n}")

    # ── 2. Lane accuracy
    print(f"\n  ┌─ 2. ACCURACY {'─'*53}")

    both_pct  = sum(f.both_lanes for f in frames) / n * 100
    ghost_pct = sum(f.is_ghost   for f in frames) / n * 100
    avg_conf  = np.mean([f.lane_confidence for f in frames])
    avg_ed    = np.mean([f.edge_density    for f in frames]) * 100

    print(f"  │  [Lane]")
    print(f"  │    Both-lane detection rate : {both_pct:6.1f}%")
    print(f"  │    Ghost-mode rate          : {ghost_pct:6.1f}%  (smoother coasting)")
    print(f"  │    Avg lane confidence      : {avg_conf:6.1f} / 100")
    print(f"  │    Avg edge density (ROI)   : {avg_ed:6.2f}%")

    frames_with_dets = sum(1 for f in frames if f.n_detections > 0)
    det_rate         = frames_with_dets / n * 100
    avg_dets         = np.mean([f.n_detections for f in frames])

    class_counts = defaultdict(int)
    prox_counts  = defaultdict(int)
    for f in frames:
        for lbl in f.det_classes:      class_counts[lbl] += 1
        for prx in f.det_proximities:  prox_counts[prx]  += 1

    total_dets = sum(class_counts.values())

    print(f"  │  [YOLO — Front Camera]")
    print(f"  │    Frames with ≥1 detection : {det_rate:6.1f}%  ({frames_with_dets}/{n})")
    print(f"  │    Avg detections / frame   : {avg_dets:6.2f}")
    print(f"  │    Total objects detected   : {total_dets}")

    if class_counts:
        print(f"  │    Class breakdown:")
        for lbl, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total_dets * 100
            bar = "█" * int(pct / 5)
            print(f"  │      {lbl:<18} {cnt:5d}  ({pct:5.1f}%)  {bar}")
    else:
        print(f"  │    Class breakdown         : (no objects detected)")

    if prox_counts:
        print(f"  │    Proximity distribution:")
        for prx in ["CLOSE", "MEDIUM", "FAR"]:
            cnt = prox_counts.get(prx, 0)
            pct = cnt / max(total_dets, 1) * 100
            print(f"  │      {prx:<8}  {cnt:5d}  ({pct:5.1f}%)")

    # ── 3. Robustness ─
    print(f"\n  ┌─ 3. ROBUSTNESS {'─'*51}")

    offsets      = [f.lane_offset for f in frames if f.lane_offset is not None]
    offset_std   = float(np.std(offsets))   if offsets else 0.0
    offset_range = (float(np.min(offsets)), float(np.max(offsets))) if offsets else (0, 0)
    jitter       = float(np.mean(np.abs(np.diff(offsets)))) if len(offsets) >= 2 else 0.0

    MAX_JITTER = 30.0
    MAX_STD    = 50.0
    stab = max(0.0, 1.0 - jitter / MAX_JITTER) * 60 + \
           max(0.0, 1.0 - offset_std / MAX_STD) * 40
    stab = round(stab, 1)

    print(f"  │  [Lane Stability]")
    print(f"  │    Offset std dev            : {offset_std:6.1f} px")
    print(f"  │    Offset range              : [{offset_range[0]:+.0f} px … {offset_range[1]:+.0f} px]")
    print(f"  │    Frame-to-frame jitter     : {jitter:6.1f} px")
    print(f"  │    Stability score           : {stab:6.1f} / 100")

    det_counts        = [f.n_detections for f in frames]
    det_std           = float(np.std(det_counts))
    yolo_consistency  = round(max(0.0, 1.0 - det_std / 5.0) * 100, 1)
    flips             = sum(
        1 for i in range(1, len(frames))
        if (frames[i].n_detections > 0) != (frames[i-1].n_detections > 0)
    )
    flip_rate = flips / max(n - 1, 1) * 100

    print(f"  │  [YOLO Consistency]")
    print(f"  │    Detection count std dev   : {det_std:6.2f}")
    print(f"  │    Detection consistency     : {yolo_consistency:6.1f} / 100")
    print(f"  │    Detect↔No-detect flips    : {flips:5d}  ({flip_rate:.1f}% of transitions)")

    # ── 4. Dual-camera command distribution ───────────────────────────────────
    print(f"\n  ┌─ 4. DUAL-CAMERA COMMAND DISTRIBUTION {'─'*29}")

    cmd_counts = defaultdict(int)
    for f in frames:
        cmd_counts[f.command] += 1

    for cmd in ["FORWARD", "STOP", "TURN_LEFT", "TURN_RIGHT", "REVERSE"]:
        cnt = cmd_counts.get(cmd, 0)
        pct = cnt / n * 100
        bar = "█" * int(pct / 5)
        print(f"  │    {cmd:<14} {cnt:5d} frames  ({pct:5.1f}%)  {bar}")

    reverse_cnt  = cmd_counts.get("REVERSE", 0)
    reverse_pct  = reverse_cnt / n * 100
    stop_cnt     = cmd_counts.get("STOP", 0)
    stop_pct     = stop_cnt / n * 100

    print(f"  │")
    print(f"  │  REVERSE triggered on {reverse_pct:.1f}% of frames  "
          f"(front blocked, rear clear)")
    print(f"  │  STOP held on        {stop_pct:.1f}% of frames  "
          f"(front & rear both blocked)")

    # ── Overall score ─────────────────────────────────────────────────────────
    print(f"\n  {SEP2}")
    overall = round(avg_conf * 0.35 + stab * 0.35 + yolo_consistency * 0.30, 1)
    print(f"  OVERALL SYSTEM SCORE : {overall:.1f} / 100")
    print(f"  (lane confidence 35% + lane stability 35% + YOLO consistency 30%)")
    print(f"{SEP}\n")
