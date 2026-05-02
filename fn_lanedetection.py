import cv2
import numpy as np
import time

_prev_centers      = None
_kmeans_seg_cache  = None
_kmeans_frame_cnt  = 0
KMEANS_STRIDE      = 3

_roi_mask_cache    = {}
_roi_region_cache  = {}

_cam_angle_cache   = 'normal'
_cam_angle_counter = 0
ANGLE_CACHE_FRAMES = 30

_GRAYS = np.array([1, 64, 128, 255], dtype=np.uint8)

def region_of_interest(img):

    h, w = img.shape[:2]
    key  = (h, w)
    if key not in _roi_region_cache:
        verts = np.array([[
            (0,          h),
            (2 * w // 5, h // 2),
            (3 * w // 5, h // 2),
            (w,          h)
        ]], dtype=np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, verts, 255)
        _roi_region_cache[key] = m
    mask = _roi_region_cache[key]
    if img.ndim == 2:
        return cv2.bitwise_and(img, mask)

    return cv2.bitwise_and(img, img, mask=mask)


def apply_roi_edges(edge_img, h, w, tight=False):

    key = (h, w, tight)
    if key not in _roi_mask_cache:
        pad = 0.08 if tight else 0.02
        verts = np.array([[
            (int(w * pad),     h),
            (int(w * pad),     int(h * 0.55)),
            (int(w * (1-pad)), int(h * 0.55)),
            (int(w * (1-pad)), h)
        ]], dtype=np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, verts, 255)
        _roi_mask_cache[key] = m
    return cv2.bitwise_and(edge_img, _roi_mask_cache[key])


def fill_cracks(mask, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def extract_lane_color_mask(image):
    hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_y2  = np.array([10,  4,  80], np.uint8)
    high_y2 = np.array([35, 200, 220], np.uint8)
    low_w   = np.array([140,  0, 220], np.uint8)
    high_w  = np.array([160,  6, 250], np.uint8)

    m_y  = cv2.inRange(hsv, low_y2, high_y2)
    m_w  = cv2.inRange(hsv, low_w,  high_w)
    mask = cv2.bitwise_or(m_y, m_w)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edges = cv2.Canny(mask, 100, 200)

    h        = image.shape[0]
    has_lane = np.count_nonzero(edges[h * 2 // 3:, :]) > 300
    return edges, has_lane

def kmeans_segmentation(img, k=4):
    global _prev_centers, _kmeans_seg_cache, _kmeans_frame_cnt

    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    _kmeans_frame_cnt += 1
    if _kmeans_seg_cache is not None and _kmeans_frame_cnt % KMEANS_STRIDE != 0:
        return _kmeans_seg_cache

    blur = cv2.GaussianBlur(small, (9, 9), 0)   # was (19,19)
    Z    = blur.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    if _prev_centers is not None:
        diffs       = Z[:, None, :] - _prev_centers[None, :, :]
        labels_init = np.argmin(np.linalg.norm(diffs, axis=2), axis=1).astype(np.int32)
        _, labels, centers = cv2.kmeans(
            Z, k, labels_init, crit, 1,
            cv2.KMEANS_USE_INITIAL_LABELS, _prev_centers)
    else:
        _, labels, centers = cv2.kmeans(
            Z, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)

    _prev_centers = centers

    gray_small = _GRAYS[labels.flatten()].reshape(blur.shape[:2])

    seg = cv2.resize(gray_small, (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    _kmeans_seg_cache = seg
    return seg


def best_label(mask_img):

    roi  = region_of_interest(mask_img)
    r, c = roi.shape
    minA = 0.05 * r * c
    scores = {}
    for v in [1, 64, 128, 255]:
        b    = (roi == v).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tot = 0
        for cnt in cnts:
            A = cv2.contourArea(cnt)
            if A < minA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h)
            if 0.2 < ar < 5.0:
                tot += A
        scores[v] = tot
    return max(scores, key=scores.get)


def create_road_mask(mask_img, label):

    bin_img     = (mask_img == label).astype(np.uint8)
    num, labels = cv2.connectedComponents(bin_img, connectivity=8)
    if num <= 1:
        return np.zeros_like(bin_img)
    best = max(range(1, num), key=lambda l: np.sum(labels == l))
    return (labels == best).astype(np.uint8) * 255

def detect_camera_angle(edge_img, h, w):
    global _cam_angle_cache, _cam_angle_counter

    _cam_angle_counter += 1
    if _cam_angle_counter % ANGLE_CACHE_FRAMES != 1:
        return _cam_angle_cache

    lines = cv2.HoughLinesP(
        edge_img, 1, np.pi / 180,
        threshold=15, minLineLength=20, maxLineGap=120
    )
    if lines is None:
        _cam_angle_cache = 'normal'
        return _cam_angle_cache

    slopes = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            continue
        m = abs((y2 - y1) / (x2 - x1))
        if 0.01 < m < 20:
            slopes.append(m)

    if not slopes:
        _cam_angle_cache = 'normal'
        return _cam_angle_cache

    _cam_angle_cache = 'low' if float(np.median(slopes)) < 0.30 else 'normal'
    return _cam_angle_cache

def classify_lines_normal(lines, img_w, img_h):
    left_pts, right_pts = [], []
    if lines is None:
        return left_pts, right_pts

    mid_x    = img_w // 2
    y_cutoff = int(img_h * 0.55)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 < y_cutoff and y2 < y_cutoff:
            continue
        if x2 == x1:
            (left_pts if x1 < mid_x else right_pts).extend([(x1, y1), (x2, y2)])
            continue
        slope = (y2 - y1) / (x2 - x1)
        cx    = (x1 + x2) / 2
        if abs(slope) < 0.35 or abs(slope) > 8.0:
            continue
        if slope < 0 and cx < mid_x:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and cx > mid_x:
            right_pts.extend([(x1, y1), (x2, y2)])

    return left_pts, right_pts

def classify_lines_low_angle(lines, img_w, img_h):
    left_pts, right_pts = [], []
    if lines is None:
        return left_pts, right_pts

    left_zone  = img_w * 0.40
    right_zone = img_w * 0.60
    y_cutoff   = int(img_h * 0.55)
    BORDER     = int(img_w * 0.08)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < BORDER and x2 < BORDER:
            continue
        if x1 > img_w - BORDER and x2 > img_w - BORDER:
            continue
        if y1 < y_cutoff and y2 < y_cutoff:
            continue
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        cx    = (x1 + x2) / 2
        if abs(slope) < 0.010 or abs(slope) > 0.60:
            continue
        if cx < left_zone:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif cx > right_zone:
            right_pts.extend([(x1, y1), (x2, y2)])

    return left_pts, right_pts


def fit_lane_line(pts, y_bottom, y_top):
    if len(pts) < 6:
        return None
    try:
        xs     = np.array([p[0] for p in pts], np.float32)
        ys     = np.array([p[1] for p in pts], np.float32)
        coeffs = np.polyfit(ys, xs, 1)
        poly   = np.poly1d(coeffs)
        return (int(poly(y_bottom)), y_bottom), (int(poly(y_top)), y_top)
    except Exception:
        return None


def estimate_ghost_from_vp(edge_img, img_w, img_h):
    lines = cv2.HoughLinesP(
        edge_img, 1, np.pi / 180,
        threshold=15, minLineLength=20, maxLineGap=120
    )
    if lines is None:
        return None, None

    left_slopes, left_intercepts   = [], []
    right_slopes, right_intercepts = [], []
    mid_x = img_w / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        m  = (y2 - y1) / (x2 - x1)
        b  = y1 - m * x1
        cx = (x1 + x2) / 2
        if abs(m) < 0.01 or abs(m) > 15:
            continue
        if m < 0 and cx < mid_x:
            left_slopes.append(m);  left_intercepts.append(b)
        elif m > 0 and cx > mid_x:
            right_slopes.append(m); right_intercepts.append(b)

    y_bottom = img_h
    y_top    = int(img_h * 0.55)

    def make_line(slopes, intercepts):
        if not slopes:
            return None
        m = float(np.median(slopes))
        b = float(np.median(intercepts))
        if abs(m) < 1e-6:
            return None
        return (int((y_bottom - b) / m), y_bottom), (int((y_top - b) / m), y_top)

    return (make_line(left_slopes,  left_intercepts),
            make_line(right_slopes, right_intercepts))


class LaneSmoother:
    def __init__(self, n=15):
        self.n            = n
        self.left_buf     = []
        self.right_buf    = []
        self.last_left    = None
        self.last_right   = None
        self.ghost_frames = 0

    def update(self, left_line, right_line, is_ghost=False):
        if is_ghost:
            self.ghost_frames += 1
            return
        self.ghost_frames = 0
        if left_line:
            self.left_buf.append(left_line)
            if len(self.left_buf) > self.n:
                self.left_buf.pop(0)
            self.last_left = left_line
        if right_line:
            self.right_buf.append(right_line)
            if len(self.right_buf) > self.n:
                self.right_buf.pop(0)
            self.last_right = right_line

    def get_smooth(self):
        def avg(buf, fallback):
            if not buf:
                return fallback
            bx = int(np.mean([l[0][0] for l in buf]))
            by = int(np.mean([l[0][1] for l in buf]))
            tx = int(np.mean([l[1][0] for l in buf]))
            ty = int(np.mean([l[1][1] for l in buf]))
            return (bx, by), (tx, ty)
        return avg(self.left_buf, self.last_left), avg(self.right_buf, self.last_right)


def draw_output(frame, left_line, right_line, alpha_fill=0.35, is_ghost=False):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cx      = None

    line_color = (0, 255, 0)
    line_w     = 6

    if left_line:
        cv2.line(overlay, left_line[0], left_line[1], line_color, line_w)
    if right_line:
        cv2.line(overlay, right_line[0], right_line[1], line_color, line_w)

    if left_line and right_line:
        poly_pts = np.array([
            left_line[0], left_line[1],
            right_line[1], right_line[0]
        ], np.int32)
        cv2.fillPoly(overlay, [poly_pts], (0, 180, 0))
        cx = (left_line[0][0] + right_line[0][0]) // 2
        cv2.line(overlay, (cx, h), (cx, int(h * 0.55)), (0, 0, 255), 3)

    result = cv2.addWeighted(frame, 1.0 - alpha_fill, overlay, alpha_fill, 0)

    if cx is not None:
        offset = cx - w // 2
        if   offset < -40: direction, color = "left",     (0,   0, 255)
        elif offset >  40: direction, color = "right",    (0,   0, 255)
        else:              direction, color = "STRAIGHT", (0, 255,   0)
        label = f"Offset: {offset}px  |  {direction}"
        if is_ghost:
            label += "  "
        cv2.putText(result, label,
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    else:
        cv2.putText(result, "Seeking lane...",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 80), 2)

    return result, cx


def run_hough_and_draw(edge_img, orig_img, smoother):
    h, w     = orig_img.shape[:2]
    y_bottom = h
    y_top    = int(h * 0.55)

    roi_quick = apply_roi_edges(edge_img, h, w, tight=True)
    cam_angle = detect_camera_angle(roi_quick, w, h)
    roi_edges = apply_roi_edges(edge_img, h, w, tight=(cam_angle == 'low'))

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=35,
        maxLineGap=100
    )

    raw_debug = orig_img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(raw_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if cam_angle == 'low':
        left_pts, right_pts = classify_lines_low_angle(lines, w, h)
    else:
        left_pts, right_pts = classify_lines_normal(lines, w, h)
        left_pts  = [p for p in left_pts  if p[0] < w * 0.50]
        right_pts = [p for p in right_pts if p[0] > w * 0.50]

    left_line  = fit_lane_line(left_pts,  y_bottom, y_top)
    right_line = fit_lane_line(right_pts, y_bottom, y_top)

    def slope(line):
        (x1, y1), (x2, y2) = line
        return (y2 - y1) / (x2 - x1 + 1e-6)

    if cam_angle == 'normal':
        if left_line  and not (-5.5 < slope(left_line)  < -0.4): left_line  = None
        if right_line and not ( 0.4 < slope(right_line) <  5.5): right_line = None
    else:
        if left_line  and not (-1.5 < slope(left_line)  < -0.005): left_line  = None
        if right_line and not ( 0.005 < slope(right_line) <  1.5): right_line = None

    confident = (left_line is not None) or (right_line is not None)

    if confident:
        smoother.update(left_line, right_line, is_ghost=False)
        smooth_left, smooth_right = smoother.get_smooth()
        result, cx = draw_output(orig_img, smooth_left, smooth_right,
                                  alpha_fill=0.35, is_ghost=False)
        mode = "CONFIDENT"
    else:
        smoother.update(None, None, is_ghost=True)
        smooth_left, smooth_right = smoother.get_smooth()
        if smooth_left is None and smooth_right is None:
            smooth_left, smooth_right = estimate_ghost_from_vp(roi_edges, w, h)
            mode = "-VP"
        else:
            mode = "-HOLD"
        ghost_alpha = max(0.10, 0.28 - smoother.ghost_frames * 0.006)
        result, cx  = draw_output(orig_img, smooth_left, smooth_right,
                                   alpha_fill=ghost_alpha, is_ghost=True)

    return result, roi_edges, raw_debug, mode

class FPSCounter:
    def __init__(self):
        self._t  = time.perf_counter()
        self.fps = 0.0

    def tick(self):
        now      = time.perf_counter()
        elapsed  = now - self._t
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._t  = now

    def draw(self, frame):
        cv2.putText(frame, "",
                    (frame.shape[1] - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)


def process_video(path):
    global _prev_centers, _kmeans_seg_cache, _kmeans_frame_cnt
    global _cam_angle_cache, _cam_angle_counter
    global _roi_mask_cache, _roi_region_cache

    _prev_centers      = None
    _kmeans_seg_cache  = None
    _kmeans_frame_cnt  = 0
    _cam_angle_cache   = 'normal'
    _cam_angle_counter = 0
    _roi_mask_cache    = {}
    _roi_region_cache  = {}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {path}")
        return

    smoother = LaneSmoother(n=15)
    fps_ctr  = FPSCounter()
    print(f"Processing: {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        edges, has_lane = extract_lane_color_mask(img)

        if has_lane:
            combined_edges = edges
        else:
            seg            = kmeans_segmentation(img)
            best_lbl       = best_label(seg)
            road_mask      = create_road_mask(seg, best_lbl)
            road_mask      = fill_cracks(road_mask)
            combined_edges = cv2.Canny(road_mask, 50, 150)

        result, roi_edges, raw_debug, mode = run_hough_and_draw(
            combined_edges, img, smoother
        )

        fps_ctr.tick()
        fps_ctr.draw(result)

        h, w         = img.shape[:2]
        gray         = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cols         = np.where(gray > 0)[1]
        lane_center  = int(np.mean(cols)) if len(cols) > 0 else None
        frame_center = w // 2
        offset       = (lane_center - frame_center) if lane_center is not None else None
        is_ghost     = mode.startswith("GHOST")

        yield {
            "frame"      : img,
            "lane_frame" : result,
            "edges"      : combined_edges,
            "roi_edges"  : roi_edges,
            "raw_debug"  : raw_debug,
            "lane_center": lane_center,
            "offset"     : offset,
            "fps"        : fps_ctr.fps,
            "mode"       : mode,
            "is_ghost"   : is_ghost,
        }

    cap.release()


def main():
    videos = [
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_043754655.TS.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_043922504.TS.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044505516.TS.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044603023.TS.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_044746327.TS.mp4",
        r"C:\Users\rzrid\Desktop\DIP\Project\Dataset\Videos\PXL_20250325_045117252.TS.mp4",
    ]
    for vid in videos:
        for pkg in process_video(vid):
            cv2.imshow("Lane", pkg["lane_frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    print("All videos processed.")


if __name__ == "__main__":
    main()