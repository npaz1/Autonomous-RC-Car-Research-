"""
RC Car YOLO V4.6 + Live LiDAR Dashboard
=====================================

Purpose
-------
First real sensor-integration dashboard:
- ESP32 camera stream over Wi-Fi
- YOLO V4.6 detections
- LD19 360-degree LiDAR scan from the ESP32-P4 HTTP API
- Manual WASD driving
- Arm/disarm + emergency stop
- Optional failure-image capture for future YOLO iterations

This version supports both MANUAL control and DWA STEERING ASSIST.

In DWA STEERING ASSIST:
- the user controls forward throttle with W
- semantic DWA controls steering
- raw LiDAR geometry remains the hard collision-safety layer
- if no safe LiDAR trajectory exists, throttle is forced to neutral
- if LiDAR data is stale/lost, throttle is forced to neutral
- YOLO-missed obstacles are still avoided because LiDAR geometry is always used

Switch back to MANUAL mode if you intentionally need to override the planner.

ESP32 firmware expected
-----------------------
ESP32-P4 simple_video_server firmware with:
- /stream
- /api/lidar_scan
- /api/lidar_status
- UDP RC control on port 4210

Controls
--------
R       Arm
T       Disarm
SPACE   Emergency stop
M       Manual mode
V       DWA steering-assist mode
W/S     Manual throttle (W forward in assist mode)
A/D     Manual steering in manual mode

P       Save missed-object frame
L       Save wrong-label frame
F       Save false-detection frame
G       Save general useful frame

Q       Quit
"""

from __future__ import annotations

import csv
import json
import math
import socket
import urllib.error
import urllib.request
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import keyboard
import numpy as np
from ultralytics import YOLO


# ============================================================================
# NETWORK
# ============================================================================

ESP32_IP = "192.168.4.1"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
LIDAR_SCAN_URL = f"http://{ESP32_IP}/api/lidar_scan"
LIDAR_STATUS_URL = f"http://{ESP32_IP}/api/lidar_status"

CONTROL_UDP_PORT = 4210

LIDAR_HTTP_TIMEOUT_S = 0.35
LIDAR_POLL_HZ = 10.0
LIDAR_POLL_PERIOD_S = 1.0 / LIDAR_POLL_HZ
LIDAR_STATUS_POLL_S = 1.0

# Dedicated safety-critical control heartbeat.
CONTROL_TX_HZ = 25.0
CONTROL_TX_PERIOD_S = 1.0 / CONTROL_TX_HZ


# ============================================================================
# RC COMMANDS
# ============================================================================

STEER_CENTER = 1500
STEER_LEFT = 1800
STEER_RIGHT = 1200

ESC_NEUTRAL = 1500
ESC_FORWARD = 1550
ESC_REVERSE = 1450


# ============================================================================
# YOLO
# ============================================================================

MODEL_PATH = Path(
    r"runs\detect\rc_car_yolo_v2\weights\best.pt"
)

YOLO_IMGSZ = 320
YOLO_CONF = 0.35

# Class-specific post-processing.
# Bush is intentionally stricter because the current model can confuse
# grass/ground texture with a bush.
CLASS_MIN_CONF = {
    "person": 0.35,
    "tree_trunk": 0.40,
    "rock": 0.45,
    "log_branch": 0.45,
    "traffic_cone": 0.40,
    "bush": 0.70,
}

# Reject bush boxes that look like broad ground-texture detections.
BUSH_MAX_FRAME_AREA_FRACTION = 0.42
BUSH_MAX_BOX_WIDTH_FRACTION = 0.80
BUSH_BOTTOM_ONLY_TOP_FRACTION = 0.62

# A bush is allowed into the semantic planner only if LiDAR confirms a
# physical obstacle at the corresponding image bearing.
BUSH_REQUIRE_LIDAR_MATCH = True

YOLO_TARGET_HZ = 5.0
YOLO_PERIOD_S = 1.0 / YOLO_TARGET_HZ
YOLO_MAX_RESULT_AGE_S = 0.75
CAMERA_STALE_S = 1.0


# ============================================================================
# LIDAR
# ============================================================================

LIDAR_BIN_DEG = 5
LIDAR_BIN_COUNT = 72

# IMPORTANT:
# This is the raw LD19 angle that physically points straight forward on the car.
# Start at zero. We will calibrate this after the dashboard works.
LIDAR_FORWARD_OFFSET_DEG = 0.0

LIDAR_MAX_DISPLAY_M = 4.0

# ============================================================================
# YOLO + LIDAR SENSOR FUSION
# ============================================================================

CAMERA_HFOV_DEG = 65.0

FUSION_CLASSES = {
    "tree_trunk",
    "rock",
    "log_branch",
    "bush",
    "person",
    "traffic_cone",
}

FUSION_BBOX_MARGIN_DEG = 3.0
FUSION_MIN_DISTANCE_M = 0.15
FUSION_MAX_DISTANCE_M = 4.0


# ============================================================================
# LIVE SEMANTIC DWA + STEERING ASSIST
# ============================================================================
#
# The planner is always visualized. It gains steering authority ONLY when the
# user explicitly selects DWA STEERING ASSIST mode. Throttle remains user-held
# and is hard-gated by LiDAR safety.
#
# Vehicle measurements collected from the physical RC car:
#   wheelbase: ~12 in = 0.3048 m
#   width: ~13.25 in = 0.3366 m
#   minimum practical grass speed: ~1.25 m/s at PWM 1600
#   measured minimum turning radius: ~1.14-1.15 m
#
# Steering measurements:
#   PWM 1200 -> radius ~1.14 m
#   PWM 1300 -> radius ~1.70 m
#   PWM 1500 -> approximately straight
#   PWM 1700 -> radius ~1.79 m
#   PWM 1800 -> radius ~1.15 m
# ============================================================================

DWA_SHADOW_ONLY = False

DWA_WHEELBASE_M = 0.3048
DWA_VEHICLE_WIDTH_M = 0.3366

DWA_FORWARD_SPEED_MPS = 1.25
DWA_FORWARD_PWM = 1600
DWA_STOP_PWM = 1500

# Full autonomous Semantic-DWA mode.
AUTO_FORWARD_PWM = 1600
AUTO_ENTRY_DELAY_S = 2.0

DWA_MEASURED_STEERING_PWM = np.array(
    [1200, 1300, 1500, 1700, 1800],
    dtype=float,
)

DWA_MEASURED_STEERING_ANGLE_DEG = np.array(
    [
        -math.degrees(math.atan(DWA_WHEELBASE_M / 1.14)),
        -math.degrees(math.atan(DWA_WHEELBASE_M / 1.70)),
        0.0,
        +math.degrees(math.atan(DWA_WHEELBASE_M / 1.79)),
        +math.degrees(math.atan(DWA_WHEELBASE_M / 1.15)),
    ],
    dtype=float,
)

# Interpolate realistic servo commands between measured points.
DWA_STEERING_PWM_SAMPLES = np.linspace(
    1200.0,
    1800.0,
    31,
)

DWA_PREDICTION_HORIZON_S = 0.80
DWA_SIMULATION_DT_S = 0.05

# Straight-ahead local navigation objective.
# This is not a global GPS goal. It simply tells the local planner:
# "continue generally forward while choosing the safest corridor."
DWA_LOCAL_GOAL_DISTANCE_M = 3.5

# Conservative point-obstacle clearance radius for this first physical test.
DWA_COLLISION_RADIUS_M = 0.23
DWA_CLEARANCE_FULL_SCORE_M = 0.80

# Ignore LiDAR returns that are almost certainly the RC car, LiDAR mount,
# wiring, or bodywork. The LD19 sits on the vehicle, so a 360-degree scan can
# see parts of the car itself.
DWA_SELF_FILTER_FORWARD_M = 0.42
DWA_SELF_FILTER_REAR_M = 0.28
DWA_SELF_FILTER_HALF_WIDTH_M = 0.28

# Steering-assist hard safety gates.
ASSIST_LIDAR_MAX_AGE_S = 0.50
ASSIST_REQUIRE_SAFE_TRAJECTORY = True

# Extended LiDAR look-ahead beyond the short DWA motion trajectory.
DWA_FORWARD_LOOKAHEAD_MAX_M = 4.0
DWA_FORWARD_LOOKAHEAD_HALF_ANGLE_DEG = 12.0

DWA_CLASS_RISK = {
    "unknown": 0.90,
    "person": 1.00,
    "tree_trunk": 0.92,
    "rock": 0.82,
    "log_branch": 0.72,
    "bush": 0.58,
    "traffic_cone": 0.65,
}

DWA_CLASS_INFLUENCE_M = {
    "unknown": 0.90,
    "person": 1.10,
    "tree_trunk": 0.78,
    "rock": 0.65,
    "log_branch": 0.75,
    "bush": 0.72,
    "traffic_cone": 0.62,
}

# Scoring weights.
DWA_W_GOAL_PROGRESS = 3.5
DWA_W_GOAL_HEADING = 1.6
DWA_W_CLEARANCE = 2.0
DWA_W_FORWARD_OPENNESS = 2.8
DWA_W_SEMANTIC_RISK = 3.0
DWA_W_STEERING = 0.35


# Live semantic-planner experiment mode.
# Keep PHYSICAL_DWA_CONTROL_ENABLED False for today's stationary validation.
PHYSICAL_DWA_CONTROL_ENABLED = False

# For the visualization, draw class-dependent semantic influence around fused objects.
SEMANTIC_FIELD_DRAW_MIN_RISK = 0.10
SEMANTIC_FIELD_MAX_RADIUS_M = 1.50


latest_lidar_distances = np.zeros(
    LIDAR_BIN_COUNT,
    dtype=np.uint16,
)

latest_lidar_sequence = 0
latest_lidar_scan_rate_hz = 0.0
latest_lidar_time = 0.0
latest_lidar_http_scans = 0
latest_lidar_http_errors = 0
latest_lidar_status = {}

lidar_lock = threading.Lock()
lidar_running = True


# ============================================================================
# CAPTURE
# ============================================================================

CAPTURE_ROOT = (
    Path("rc_yolo_dataset")
    / "iteration_3_collection"
)

CAPTURE_FOLDERS = {
    "missed": CAPTURE_ROOT / "missed_objects",
    "wrong_label": CAPTURE_ROOT / "wrong_labels",
    "false_detection": CAPTURE_ROOT / "false_detections",
    "general": CAPTURE_ROOT / "general_new_examples",
}

for folder in CAPTURE_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

CAPTURE_LOG = CAPTURE_ROOT / "capture_log.csv"


# ============================================================================
# DISPLAY
# ============================================================================

WINDOW_NAME = "RC Car | YOLO V4.6 + LD19 LiDAR"

CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600

LIDAR_VIEW_SIZE = 600
STATUS_HEIGHT = 70

WINDOW_WIDTH = (
    CAMERA_WIDTH
    + LIDAR_VIEW_SIZE
)

WINDOW_HEIGHT = CAMERA_HEIGHT + STATUS_HEIGHT


# ============================================================================
# SOCKETS
# ============================================================================

# UDP is now used only for RC steering/throttle/arm control.
# LiDAR is fetched independently from the ESP32-P4 HTTP API.
control_sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)


# ============================================================================
# INDEPENDENT CONTROL HEARTBEAT STATE
# ============================================================================

command_lock = threading.Lock()

desired_steering_us = STEER_CENTER
desired_throttle_us = ESC_NEUTRAL
desired_armed = False

control_thread_running = True

control_tx_count = 0
control_tx_rate_hz = 0.0
control_tx_last_send_time = 0.0
control_tx_max_gap_s = 0.0
control_tx_last_gap_s = 0.0
control_tx_diag_window_start = time.time()
control_tx_diag_window_count = 0


# ============================================================================
# CONTROL
# ============================================================================

def send_command_raw(
    steering_us: int,
    throttle_us: int,
    armed: bool,
) -> None:
    """
    Current ESP32-P4 firmware expects:
        CTRL:<steering_us>,<throttle_us>

    Arm/disarm is enforced locally by the dashboard. When disarmed, every
    heartbeat sent to the car is forced to center steering + neutral throttle.
    """
    if not armed:
        steering_us = STEER_CENTER
        throttle_us = ESC_NEUTRAL

    command = (
        f"CTRL:{int(steering_us)},"
        f"{int(throttle_us)}"
    )

    try:
        control_sock.sendto(
            command.encode("utf-8"),
            (
                ESP32_IP,
                CONTROL_UDP_PORT,
            ),
        )
    except OSError as error:
        print(f"Control UDP error: {error}")


def set_desired_command(
    steering_us: int,
    throttle_us: int,
    armed: bool,
) -> None:
    global desired_steering_us
    global desired_throttle_us
    global desired_armed

    with command_lock:
        desired_steering_us = int(steering_us)
        desired_throttle_us = int(throttle_us)
        desired_armed = bool(armed)


def get_desired_command() -> tuple[int, int, bool]:
    with command_lock:
        return (
            int(desired_steering_us),
            int(desired_throttle_us),
            bool(desired_armed),
        )


def emergency_stop() -> None:
    set_desired_command(
        STEER_CENTER,
        ESC_NEUTRAL,
        False,
    )

    send_command_raw(
        STEER_CENTER,
        ESC_NEUTRAL,
        False,
    )

    time.sleep(0.02)

    send_command_raw(
        STEER_CENTER,
        ESC_NEUTRAL,
        False,
    )


def control_tx_worker() -> None:
    global control_thread_running
    global control_tx_count
    global control_tx_rate_hz
    global control_tx_last_send_time
    global control_tx_max_gap_s
    global control_tx_last_gap_s
    global control_tx_diag_window_start
    global control_tx_diag_window_count

    next_send_time = time.perf_counter()

    while control_thread_running:
        now_perf = time.perf_counter()

        if now_perf < next_send_time:
            time.sleep(
                min(
                    next_send_time - now_perf,
                    0.005,
                )
            )
            continue

        steering_us, throttle_us, armed = get_desired_command()

        wall_now = time.time()

        if control_tx_last_send_time > 0.0:
            gap_s = wall_now - control_tx_last_send_time
            control_tx_last_gap_s = gap_s

            if gap_s > control_tx_max_gap_s:
                control_tx_max_gap_s = gap_s

        send_command_raw(
            steering_us,
            throttle_us,
            armed,
        )

        control_tx_last_send_time = wall_now
        control_tx_count += 1
        control_tx_diag_window_count += 1

        elapsed_window = wall_now - control_tx_diag_window_start

        if elapsed_window >= 1.0:
            control_tx_rate_hz = (
                control_tx_diag_window_count
                / max(elapsed_window, 0.001)
            )

            control_tx_diag_window_count = 0
            control_tx_diag_window_start = wall_now

        next_send_time += CONTROL_TX_PERIOD_S

        if (
            time.perf_counter() - next_send_time
            > CONTROL_TX_PERIOD_S
        ):
            next_send_time = (
                time.perf_counter()
                + CONTROL_TX_PERIOD_S
            )


control_tx_thread = threading.Thread(
    target=control_tx_worker,
    daemon=True,
)

control_tx_thread.start()


# ============================================================================
# LIDAR HTTP POLLING THREAD
# ============================================================================

def http_get_json(url, timeout_s=LIDAR_HTTP_TIMEOUT_S):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Connection": "close",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _distance_mm(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0

    if not math.isfinite(value) or value <= 0:
        return 0

    return int(max(0, min(65535, round(value))))


def parse_lidar_scan_payload(payload):
    """
    Normalize /api/lidar_scan into the existing 72-bin, 5-degree scan.

    Accepted layouts include:
      {"distances_mm": [72 values], ...}
      {"distances": [72 values], ...}
      {"bins": [72 values], ...}
      {"points": [{"angle_deg": 12.3, "distance_mm": 850}, ...]}

    Bare 72-value arrays and [angle, distance] point arrays are also accepted.
    """
    sequence = 0
    scan_rate_hz = 0.0
    point_list = None

    if isinstance(payload, dict):
        for key in ("sequence", "seq", "scan_sequence", "scan_id"):
            if key in payload:
                try:
                    sequence = int(payload[key])
                except (TypeError, ValueError):
                    pass
                break

        for key in ("scan_rate_hz", "rate_hz", "hz", "scan_hz"):
            if key in payload:
                try:
                    scan_rate_hz = float(payload[key])
                except (TypeError, ValueError):
                    pass
                break

        for key in ("distances_mm", "distances", "bins", "ranges_mm", "ranges"):
            values = payload.get(key)
            if (
                isinstance(values, list)
                and len(values) == LIDAR_BIN_COUNT
                and not any(isinstance(item, dict) for item in values)
            ):
                return (
                    np.asarray([_distance_mm(v) for v in values], dtype=np.uint16),
                    sequence,
                    scan_rate_hz,
                )

        for key in ("points", "scan", "samples", "measurements", "data"):
            if isinstance(payload.get(key), list):
                point_list = payload[key]
                break

    elif isinstance(payload, list):
        point_list = payload

    if point_list is None:
        raise ValueError("Unrecognized /api/lidar_scan JSON layout")

    if (
        len(point_list) == LIDAR_BIN_COUNT
        and not any(isinstance(item, dict) for item in point_list)
        and not any(isinstance(item, (list, tuple)) for item in point_list)
    ):
        return (
            np.asarray([_distance_mm(v) for v in point_list], dtype=np.uint16),
            sequence,
            scan_rate_hz,
        )

    distances = np.zeros(LIDAR_BIN_COUNT, dtype=np.uint16)
    valid_points = 0

    for item in point_list:
        angle = None
        distance = None

        if isinstance(item, dict):
            for key in ("angle_deg", "angle", "deg", "bearing_deg"):
                if key in item:
                    angle = item[key]
                    break
            for key in ("distance_mm", "distance", "range_mm", "range", "mm"):
                if key in item:
                    distance = item[key]
                    break

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            angle, distance = item[0], item[1]

        if angle is None or distance is None:
            continue

        try:
            angle = float(angle) % 360.0
        except (TypeError, ValueError):
            continue

        distance = _distance_mm(distance)
        if distance <= 0:
            continue

        bin_index = int(round(angle / LIDAR_BIN_DEG)) % LIDAR_BIN_COUNT
        old = int(distances[bin_index])

        if old == 0 or distance < old:
            distances[bin_index] = distance

        valid_points += 1

    if valid_points == 0:
        raise ValueError("LiDAR JSON contained no valid points")

    return distances, sequence, scan_rate_hz


def lidar_receiver():
    global latest_lidar_distances
    global latest_lidar_sequence
    global latest_lidar_scan_rate_hz
    global latest_lidar_time
    global latest_lidar_http_scans
    global latest_lidar_http_errors
    global latest_lidar_status
    global lidar_running

    next_scan = time.perf_counter()
    next_status = time.perf_counter()
    local_sequence = 0
    last_error_print = 0.0

    while lidar_running:
        now_perf = time.perf_counter()

        if now_perf < next_scan:
            time.sleep(min(next_scan - now_perf, 0.005))
            continue

        next_scan += LIDAR_POLL_PERIOD_S
        if time.perf_counter() - next_scan > LIDAR_POLL_PERIOD_S:
            next_scan = time.perf_counter() + LIDAR_POLL_PERIOD_S

        try:
            payload = http_get_json(LIDAR_SCAN_URL)
            distances, sequence, scan_rate_hz = parse_lidar_scan_payload(payload)

            local_sequence += 1
            if sequence == 0:
                sequence = local_sequence

            with lidar_lock:
                latest_lidar_distances = distances
                latest_lidar_sequence = sequence
                if scan_rate_hz > 0:
                    latest_lidar_scan_rate_hz = scan_rate_hz
                latest_lidar_time = time.time()
                latest_lidar_http_scans += 1

        except Exception as error:
            with lidar_lock:
                latest_lidar_http_errors += 1
                last_good_scan_time = latest_lidar_time

            now = time.time()

            scan_stale = (
                last_good_scan_time <= 0.0
                or now - last_good_scan_time >= 0.75
            )

            if scan_stale and now - last_error_print >= 2.0:
                print(
                    "LiDAR HTTP scan error while data is stale: "
                    f"{error}"
                )
                last_error_print = now

        if now_perf >= next_status:
            next_status = now_perf + LIDAR_STATUS_POLL_S

            try:
                status = http_get_json(LIDAR_STATUS_URL)

                if isinstance(status, dict):
                    with lidar_lock:
                        latest_lidar_status = dict(status)

                        for key in ("scan_rate_hz", "rate_hz", "hz"):
                            if key in status:
                                try:
                                    rate = float(status[key])
                                except (TypeError, ValueError):
                                    continue

                                if rate > 0:
                                    latest_lidar_scan_rate_hz = rate
                                break

            except Exception:
                # Status is diagnostic only. Scan freshness remains the
                # safety-critical signal.
                pass


lidar_thread = threading.Thread(
    target=lidar_receiver,
    daemon=True,
)

lidar_thread.start()


# ============================================================================
# CAPTURE HELPERS
# ============================================================================

def ensure_capture_log() -> None:
    if CAPTURE_LOG.exists():
        return

    with CAPTURE_LOG.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "category",
                "filename",
                "armed",
                "steering_us",
                "throttle_us",
                "detections_json",
            ]
        )


def serialize_boxes(boxes) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "confidence": round(float(conf), 4),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        }
        for x1, y1, x2, y2, label, conf in boxes
    ]


def save_training_frame(
    raw_frame: np.ndarray,
    category: str,
    armed: bool,
    steering_us: int,
    throttle_us: int,
    boxes,
) -> None:
    now = datetime.now()

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    filename = (
        CAPTURE_FOLDERS[category]
        / f"{category}_{timestamp}.jpg"
    )

    if not cv2.imwrite(
        str(filename),
        raw_frame,
    ):
        print(
            f"Could not save {filename}"
        )
        return

    with CAPTURE_LOG.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                now.isoformat(
                    timespec="milliseconds"
                ),
                category,
                str(filename),
                armed,
                steering_us,
                throttle_us,
                json.dumps(
                    serialize_boxes(boxes),
                    separators=(",", ":"),
                ),
            ]
        )

    print(
        f"Saved {category}: {filename}"
    )


# ============================================================================
# KEY DEBOUNCE
# ============================================================================

previous_key_state: dict[str, bool] = {}


def pressed_once(key_name: str) -> bool:
    current = keyboard.is_pressed(
        key_name
    )

    previous = previous_key_state.get(
        key_name,
        False,
    )

    previous_key_state[key_name] = current

    return current and not previous


# ============================================================================
# YOLO + LIDAR FUSION HELPERS
# ============================================================================

def wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def camera_x_to_bearing_deg(
    x_pixel: float,
    image_width: int,
) -> float:
    normalized_x = (
        float(x_pixel)
        / max(float(image_width), 1.0)
    ) - 0.5

    return normalized_x * CAMERA_HFOV_DEG


def get_lidar_match_for_box(
    left_bearing_deg: float,
    right_bearing_deg: float,
    center_bearing_deg: float,
    used_bins: set[int],
) -> tuple[float | None, float | None, int | None]:

    with lidar_lock:
        distances = latest_lidar_distances.copy()
        scan_age = (
            time.time() - latest_lidar_time
            if latest_lidar_time > 0
            else 999.0
        )

    if scan_age > 0.5:
        return None, None, None

    low = min(
        left_bearing_deg,
        right_bearing_deg,
    ) - FUSION_BBOX_MARGIN_DEG

    high = max(
        left_bearing_deg,
        right_bearing_deg,
    ) + FUSION_BBOX_MARGIN_DEG

    candidates = []

    for bin_index, distance_mm in enumerate(
        distances
    ):
        if distance_mm <= 0:
            continue

        if bin_index in used_bins:
            continue

        distance_m = float(distance_mm) / 1000.0

        if not (
            FUSION_MIN_DISTANCE_M
            <= distance_m
            <= FUSION_MAX_DISTANCE_M
        ):
            continue

        raw_angle_deg = bin_index * LIDAR_BIN_DEG

        relative_angle_deg = wrap_angle_deg(
            raw_angle_deg
            - LIDAR_FORWARD_OFFSET_DEG
        )

        if not (
            low
            <= relative_angle_deg
            <= high
        ):
            continue

        center_error_deg = abs(
            wrap_angle_deg(
                relative_angle_deg
                - center_bearing_deg
            )
        )

        score = (
            center_error_deg
            + 0.15 * distance_m
        )

        candidates.append(
            (
                score,
                distance_m,
                relative_angle_deg,
                bin_index,
            )
        )

    if not candidates:
        return None, None, None

    candidates.sort(
        key=lambda item: item[0]
    )

    _, distance_m, bearing_deg, bin_index = (
        candidates[0]
    )

    return (
        float(distance_m),
        float(bearing_deg),
        int(bin_index),
    )


def fuse_yolo_detections(
    boxes,
    image_width: int,
) -> list[dict]:

    fused_objects = []
    used_bins: set[int] = set()

    sorted_boxes = sorted(
        boxes,
        key=lambda item: float(item[5]),
        reverse=True,
    )

    for (
        x1,
        y1,
        x2,
        y2,
        label,
        confidence,
    ) in sorted_boxes:

        if label not in FUSION_CLASSES:
            continue

        left_bearing_deg = camera_x_to_bearing_deg(
            float(x1),
            image_width,
        )

        right_bearing_deg = camera_x_to_bearing_deg(
            float(x2),
            image_width,
        )

        center_x = (
            float(x1) + float(x2)
        ) / 2.0

        center_bearing_deg = camera_x_to_bearing_deg(
            center_x,
            image_width,
        )

        (
            distance_m,
            lidar_bearing_deg,
            lidar_bin,
        ) = get_lidar_match_for_box(
            left_bearing_deg,
            right_bearing_deg,
            center_bearing_deg,
            used_bins,
        )

        x_right_m = None
        y_forward_m = None

        if (
            distance_m is not None
            and lidar_bearing_deg is not None
            and lidar_bin is not None
        ):
            used_bins.add(lidar_bin)

            theta = math.radians(
                lidar_bearing_deg
            )

            x_right_m = (
                distance_m
                * math.sin(theta)
            )

            y_forward_m = (
                distance_m
                * math.cos(theta)
            )

        fused_objects.append(
            {
                "label": str(label),
                "bbox": (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ),
                "confidence": float(confidence),
                "camera_bearing_deg": float(
                    center_bearing_deg
                ),
                "distance_m": distance_m,
                "lidar_bearing_deg": (
                    lidar_bearing_deg
                ),
                "lidar_bin": lidar_bin,
                "x_right_m": x_right_m,
                "y_forward_m": y_forward_m,
            }
        )

    return fused_objects




# ============================================================================
# YOLO CLASS-SPECIFIC FALSE-POSITIVE FILTERING
# ============================================================================

def filter_yolo_boxes(
    boxes,
    frame_width: int,
    frame_height: int,
):
    """
    Apply class-specific confidence and simple image-geometry filtering.

    This does not retrain or alter the YOLO model. It only prevents weak or
    ground-like bush detections from entering sensor fusion/planning.
    """
    filtered = []

    frame_area = max(
        1.0,
        float(frame_width * frame_height),
    )

    for box in boxes:
        x1, y1, x2, y2, label, confidence = box

        label = str(label)
        confidence = float(confidence)

        min_conf = float(
            CLASS_MIN_CONF.get(
                label,
                YOLO_CONF,
            )
        )

        if confidence < min_conf:
            continue

        if label == "bush":
            box_w = max(
                0.0,
                float(x2 - x1),
            )
            box_h = max(
                0.0,
                float(y2 - y1),
            )

            area_fraction = (
                box_w * box_h
            ) / frame_area

            width_fraction = (
                box_w
                / max(
                    1.0,
                    float(frame_width),
                )
            )

            top_fraction = (
                float(y1)
                / max(
                    1.0,
                    float(frame_height),
                )
            )

            # Grass false positives tend to be broad and/or begin very low
            # in the frame. Real bushes can still extend to the bottom.
            if (
                area_fraction
                > BUSH_MAX_FRAME_AREA_FRACTION
            ):
                continue

            if (
                width_fraction
                > BUSH_MAX_BOX_WIDTH_FRACTION
            ):
                continue

            if (
                top_fraction
                > BUSH_BOTTOM_ONLY_TOP_FRACTION
            ):
                continue

        filtered.append(
            (
                x1,
                y1,
                x2,
                y2,
                label,
                confidence,
            )
        )

    return filtered


# ============================================================================
# SEMANTIC OBSTACLE REPRESENTATION
# ============================================================================

def build_semantic_obstacle_list(
    fused_objects,
) -> list[dict]:
    """
    Convert fused YOLO+LiDAR detections into the planner-facing representation.

    Coordinate convention:
      x_m > 0  -> vehicle right
      x_m < 0  -> vehicle left
      y_m > 0  -> forward
    """
    obstacles = []

    for fused_object in fused_objects:
        x_m = fused_object.get("x_right_m")
        y_m = fused_object.get("y_forward_m")
        distance_m = fused_object.get("distance_m")
        bearing_deg = fused_object.get("lidar_bearing_deg")

        if (
            x_m is None
            or y_m is None
            or distance_m is None
            or bearing_deg is None
        ):
            continue

        label = str(
            fused_object.get(
                "label",
                "unknown",
            )
        )

        confidence = float(
            fused_object.get(
                "confidence",
                0.0,
            )
        )


        if (
            label == "bush"
            and BUSH_REQUIRE_LIDAR_MATCH
            and not bool(
                fused_object.get(
                    "lidar_matched",
                    True,
                )
            )
        ):
            continue

        obstacles.append(
            {
                "label": label,
                "confidence": confidence,
                "x_m": float(x_m),
                "y_m": float(y_m),
                "distance_m": float(distance_m),
                "bearing_deg": float(bearing_deg),
                "risk": float(
                    DWA_CLASS_RISK.get(
                        label,
                        DWA_CLASS_RISK["unknown"],
                    )
                ),
                "influence_m": float(
                    DWA_CLASS_INFLUENCE_M.get(
                        label,
                        DWA_CLASS_INFLUENCE_M["unknown"],
                    )
                ),
            }
        )

    return obstacles


def semantic_field_value(
    x_m: float,
    y_m: float,
    semantic_obstacles,
) -> float:
    """
    V15-style semantic field: class-dependent Gaussian risk around each
    detected obstacle. Higher value = larger penalty.
    """
    total = 0.0

    for obstacle in semantic_obstacles:
        dx = float(x_m) - float(obstacle["x_m"])
        dy = float(y_m) - float(obstacle["y_m"])

        distance = math.hypot(dx, dy)
        influence = max(
            float(obstacle["influence_m"]),
            0.05,
        )

        class_risk = float(
            obstacle["risk"]
        )

        confidence = float(
            obstacle["confidence"]
        )

        normalized = distance / influence

        total += (
            class_risk
            * confidence
            * math.exp(
                -0.5
                * normalized
                * normalized
            )
        )

    return min(
        total,
        2.0,
    )

# ============================================================================
# LIVE SEMANTIC DWA HELPERS
# ============================================================================

def dwa_steering_angle_from_pwm(
    pwm: float,
) -> float:
    """
    Piecewise-linear interpolation through the steering measurements.
    Returns steering angle in radians.

    Sign convention:
      negative = right turn
      positive = left turn
    """
    angle_deg = float(
        np.interp(
            pwm,
            DWA_MEASURED_STEERING_PWM,
            DWA_MEASURED_STEERING_ANGLE_DEG,
        )
    )

    return math.radians(angle_deg)


def dwa_simulate_trajectory(
    steering_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bicycle/Ackermann model in the LiDAR display coordinate system.

    x = positive to vehicle right
    y = positive forward
    heading positive = vehicle rotates left
    """
    step_count = (
        int(
            round(
                DWA_PREDICTION_HORIZON_S
                / DWA_SIMULATION_DT_S
            )
        )
        + 1
    )

    x = np.zeros(
        step_count,
        dtype=float,
    )

    y = np.zeros(
        step_count,
        dtype=float,
    )

    heading = np.zeros(
        step_count,
        dtype=float,
    )

    for index in range(
        1,
        step_count,
    ):
        previous_heading = (
            heading[index - 1]
        )

        yaw_rate = (
            DWA_FORWARD_SPEED_MPS
            * math.tan(
                steering_angle_rad
            )
            / DWA_WHEELBASE_M
        )

        # Positive heading is a left turn, therefore x moves negative.
        x[index] = (
            x[index - 1]
            - DWA_FORWARD_SPEED_MPS
            * math.sin(
                previous_heading
            )
            * DWA_SIMULATION_DT_S
        )

        y[index] = (
            y[index - 1]
            + DWA_FORWARD_SPEED_MPS
            * math.cos(
                previous_heading
            )
            * DWA_SIMULATION_DT_S
        )

        heading[index] = (
            previous_heading
            + yaw_rate
            * DWA_SIMULATION_DT_S
        )

    return x, y, heading


# Grass/foliage clutter filter for the DWA HARD-COLLISION layer only.
# The raw 72-bin scan is still used by the camera/LiDAR semantic fusion.
LIDAR_COLLISION_CLUSTER_MIN_BINS = 3
LIDAR_COLLISION_CLUSTER_HALF_WINDOW_BINS = 2   # +/- 2 bins = +/- 10 deg at 5 deg/bin
LIDAR_COLLISION_CLUSTER_RANGE_TOLERANCE_M = 0.15


# Near-field grass suppression.
#
# The LD19 is mounted low enough that blades of grass can generate dense
# returns very close to the sensor. These returns can survive the cluster
# filter and cause every DWA candidate to be rejected.
#
# For the planner collision cloud, ignore returns closer than this floor.
DWA_GRASS_NEARFIELD_IGNORE_M = 0.35

# For YOLO/LiDAR semantic fusion, also reject implausibly close LiDAR matches
# so a distant person is not incorrectly paired with a nearby grass blade.
FUSION_MIN_DISTANCE_M = max(FUSION_MIN_DISTANCE_M, 0.45)

# Dense-close-object emergency guard. If enough raw returns appear very close
# in a forward sector, we still treat it as an immediate physical hazard.
DENSE_CLOSE_HAZARD_DISTANCE_M = 0.30
DENSE_CLOSE_HAZARD_MIN_BINS = 7
DENSE_CLOSE_HAZARD_FORWARD_HALF_ANGLE_DEG = 35.0


def filter_lidar_collision_xy_from_bins(
    distances,
) -> np.ndarray:
    """
    Convert the reduced LD19 scan to DWA collision points while rejecting
    isolated grass-blade returns.

    A bin survives only if at least 3 returns in its +/-2-bin angular
    neighborhood have approximately the same range (within 15 cm).

    This filter is deliberately applied ONLY to DWA geometric collision
    checking. Raw LiDAR remains untouched for YOLO/LiDAR fusion.
    """
    accepted = []

    bin_count = len(distances)

    for bin_index, distance_mm in enumerate(distances):
        if distance_mm <= 0:
            continue

        distance_m = float(distance_mm) / 1000.0

        if not (
            DWA_GRASS_NEARFIELD_IGNORE_M
            <= distance_m
            <= DWA_FORWARD_LOOKAHEAD_MAX_M
        ):
            continue

        support = 0

        for offset in range(
            -LIDAR_COLLISION_CLUSTER_HALF_WINDOW_BINS,
            LIDAR_COLLISION_CLUSTER_HALF_WINDOW_BINS + 1,
        ):
            neighbor_index = (bin_index + offset) % bin_count
            neighbor_mm = distances[neighbor_index]

            if neighbor_mm <= 0:
                continue

            neighbor_m = float(neighbor_mm) / 1000.0

            if abs(neighbor_m - distance_m) <= LIDAR_COLLISION_CLUSTER_RANGE_TOLERANCE_M:
                support += 1

        if support < LIDAR_COLLISION_CLUSTER_MIN_BINS:
            continue

        raw_angle_deg = bin_index * LIDAR_BIN_DEG
        relative_angle_deg = wrap_angle_deg(
            raw_angle_deg - LIDAR_FORWARD_OFFSET_DEG
        )
        theta = math.radians(relative_angle_deg)

        x_m = distance_m * math.sin(theta)
        y_m = distance_m * math.cos(theta)

        # Preserve the existing self-filter.
        if (
            abs(x_m) <= DWA_SELF_FILTER_HALF_WIDTH_M
            and -DWA_SELF_FILTER_REAR_M <= y_m <= DWA_SELF_FILTER_FORWARD_M
        ):
            continue

        if y_m < -0.30:
            continue

        accepted.append((x_m, y_m))

    if not accepted:
        return np.empty((0, 2), dtype=float)

    return np.asarray(accepted, dtype=float)



def has_dense_close_hazard(
    distances,
) -> bool:
    """
    Detect a genuine dense, very-close forward obstacle without letting
    scattered grass blades force DWA into safe=0.

    Requirements:
      - returns must lie in the forward sector
      - at least 7 CONSECUTIVE 5-degree bins must be occupied
      - neighboring ranges must agree within 0.10 m

    A few isolated or scattered grass returns no longer create the synthetic
    blocker used by the hard-safety layer.
    """
    candidates = []

    for bin_index, distance_mm in enumerate(distances):
        if distance_mm <= 0:
            candidates.append(None)
            continue

        distance_m = float(distance_mm) / 1000.0

        raw_angle_deg = bin_index * LIDAR_BIN_DEG
        relative_angle_deg = wrap_angle_deg(
            raw_angle_deg - LIDAR_FORWARD_OFFSET_DEG
        )

        if (
            distance_m <= DENSE_CLOSE_HAZARD_DISTANCE_M
            and abs(relative_angle_deg)
            <= DENSE_CLOSE_HAZARD_FORWARD_HALF_ANGLE_DEG
        ):
            candidates.append(distance_m)
        else:
            candidates.append(None)

    n = len(candidates)

    # Check all possible contiguous runs, including wrap-around.
    extended = candidates + candidates[:DENSE_CLOSE_HAZARD_MIN_BINS - 1]

    for start_index in range(n):
        run = extended[
            start_index:
            start_index + DENSE_CLOSE_HAZARD_MIN_BINS
        ]

        if any(value is None for value in run):
            continue

        if max(run) - min(run) <= 0.10:
            return True

    return False


def get_live_lidar_xy() -> np.ndarray:
    """
    Return the FILTERED LiDAR point cloud used by DWA hard-collision checking.

    Raw LiDAR is intentionally left unchanged elsewhere so semantic fusion
    still has access to the original scan.
    """
    with lidar_lock:
        distances = latest_lidar_distances.copy()
        scan_age = (
            time.time() - latest_lidar_time
            if latest_lidar_time > 0
            else 999.0
        )

    if scan_age > 0.5:
        return np.empty((0, 2), dtype=float)

    filtered = filter_lidar_collision_xy_from_bins(
        distances
    )

    if has_dense_close_hazard(distances):
        # Add a synthetic blocker directly ahead so DWA cannot drive forward
        # through a truly dense near-field obstacle.
        blocker = np.asarray(
            [[0.0, DENSE_CLOSE_HAZARD_DISTANCE_M]],
            dtype=float,
        )

        if filtered.size == 0:
            return blocker

        filtered = np.vstack(
            (
                filtered,
                blocker,
            )
        )

    return filtered


def dwa_minimum_clearance(
    trajectory_x: np.ndarray,
    trajectory_y: np.ndarray,
    lidar_points: np.ndarray,
) -> float:
    """
    Minimum centerline-to-LiDAR-point distance along a candidate trajectory.

    DWA_COLLISION_RADIUS_M converts this point distance into a conservative
    vehicle collision test.
    """
    if lidar_points.size == 0:
        return DWA_FORWARD_LOOKAHEAD_MAX_M

    trajectory_points = np.column_stack(
        (
            trajectory_x,
            trajectory_y,
        )
    )

    difference = (
        trajectory_points[:, None, :]
        - lidar_points[None, :, :]
    )

    distances = np.linalg.norm(
        difference,
        axis=2,
    )

    return float(
        np.min(distances)
    )


def dwa_forward_open_distance(
    candidate_bearing_deg: float,
    lidar_points: np.ndarray,
) -> float:
    """
    Look farther than the short DWA trajectory using the real LiDAR scan.

    Returns the nearest LiDAR obstacle within a narrow corridor centered on
    the candidate direction. If nothing is seen, returns the sensor limit.
    """
    if lidar_points.size == 0:
        return DWA_FORWARD_LOOKAHEAD_MAX_M

    best_distance = (
        DWA_FORWARD_LOOKAHEAD_MAX_M
    )

    for x_m, y_m in lidar_points:
        if y_m <= 0.0:
            continue

        distance_m = math.hypot(
            float(x_m),
            float(y_m),
        )

        bearing_deg = math.degrees(
            math.atan2(
                float(x_m),
                float(y_m),
            )
        )

        angle_error = abs(
            wrap_angle_deg(
                bearing_deg
                - candidate_bearing_deg
            )
        )

        if (
            angle_error
            <= DWA_FORWARD_LOOKAHEAD_HALF_ANGLE_DEG
        ):
            best_distance = min(
                best_distance,
                distance_m,
            )

    return best_distance


def dwa_semantic_risk(
    trajectory_x: np.ndarray,
    trajectory_y: np.ndarray,
    candidate_bearing_deg: float,
    semantic_obstacles,
) -> float:
    """
    V15-style semantic cost for a candidate trajectory.

    Uses the peak class-dependent semantic field value along the candidate
    trajectory, plus a mild angular look-ahead term for fused objects that are
    slightly beyond the short motion horizon.
    """
    if not semantic_obstacles:
        return 0.0

    peak_path_risk = 0.0

    for x_m, y_m in zip(
        trajectory_x,
        trajectory_y,
    ):
        peak_path_risk = max(
            peak_path_risk,
            semantic_field_value(
                float(x_m),
                float(y_m),
                semantic_obstacles,
            ),
        )

    corridor_risk = 0.0

    for obstacle in semantic_obstacles:
        angular_error = abs(
            wrap_angle_deg(
                float(obstacle["bearing_deg"])
                - candidate_bearing_deg
            )
        )

        angular_factor = max(
            0.0,
            1.0
            - angular_error / 30.0,
        )

        distance_factor = max(
            0.0,
            1.0
            - float(obstacle["distance_m"])
            / DWA_FORWARD_LOOKAHEAD_MAX_M,
        )

        corridor_risk = max(
            corridor_risk,
            float(obstacle["risk"])
            * float(obstacle["confidence"])
            * angular_factor
            * distance_factor,
        )

    return min(
        max(
            peak_path_risk,
            corridor_risk,
        ),
        2.0,
    )


def run_semantic_dwa_shadow(
    semantic_obstacles,
) -> dict:
    """
    Run one real-time semantic DWA decision using the current sensor scan.

    This function RETURNS a proposed command only.
    It does not transmit anything to the ESP32.
    """
    lidar_points = (
        get_live_lidar_xy()
    )

    candidates = []
    safe_candidates = []

    for steering_pwm in (
        DWA_STEERING_PWM_SAMPLES
    ):
        steering_angle_rad = (
            dwa_steering_angle_from_pwm(
                float(steering_pwm)
            )
        )

        (
            trajectory_x,
            trajectory_y,
            headings,
        ) = dwa_simulate_trajectory(
            steering_angle_rad
        )

        minimum_clearance = (
            dwa_minimum_clearance(
                trajectory_x,
                trajectory_y,
                lidar_points,
            )
        )

        collision = (
            minimum_clearance
            < DWA_COLLISION_RADIUS_M
        )

        endpoint_x = float(
            trajectory_x[-1]
        )

        endpoint_y = float(
            trajectory_y[-1]
        )

        endpoint_heading = float(
            headings[-1]
        )

        start_goal_distance = (
            DWA_LOCAL_GOAL_DISTANCE_M
        )

        end_goal_distance = math.hypot(
            endpoint_x,
            DWA_LOCAL_GOAL_DISTANCE_M
            - endpoint_y,
        )

        progress_m = (
            start_goal_distance
            - end_goal_distance
        )

        maximum_progress = max(
            DWA_FORWARD_SPEED_MPS
            * DWA_PREDICTION_HORIZON_S,
            0.01,
        )

        goal_progress_score = max(
            0.0,
            min(
                1.0,
                progress_m
                / maximum_progress,
            ),
        )

        desired_heading = math.atan2(
            -endpoint_x,
            DWA_LOCAL_GOAL_DISTANCE_M
            - endpoint_y,
        )

        heading_error = abs(
            math.atan2(
                math.sin(
                    desired_heading
                    - endpoint_heading
                ),
                math.cos(
                    desired_heading
                    - endpoint_heading
                ),
            )
        )

        goal_heading_score = (
            1.0
            - max(
                0.0,
                min(
                    1.0,
                    heading_error
                    / (math.pi / 2.0),
                ),
            )
        )

        clearance_score = max(
            0.0,
            min(
                1.0,
                (
                    minimum_clearance
                    - DWA_COLLISION_RADIUS_M
                )
                / max(
                    DWA_CLEARANCE_FULL_SCORE_M
                    - DWA_COLLISION_RADIUS_M,
                    0.01,
                ),
            ),
        )

        # Candidate direction in LiDAR/camera bearing convention:
        # positive = right.
        candidate_bearing_deg = math.degrees(
            math.atan2(
                endpoint_x,
                max(
                    endpoint_y,
                    0.001,
                ),
            )
        )

        forward_open_distance = (
            dwa_forward_open_distance(
                candidate_bearing_deg,
                lidar_points,
            )
        )

        forward_openness_score = max(
            0.0,
            min(
                1.0,
                forward_open_distance
                / DWA_FORWARD_LOOKAHEAD_MAX_M,
            ),
        )

        semantic_risk = (
            dwa_semantic_risk(
                trajectory_x,
                trajectory_y,
                candidate_bearing_deg,
                semantic_obstacles,
            )
        )

        steering_penalty = (
            abs(
                float(steering_pwm)
                - STEER_CENTER
            )
            / max(
                float(
                    STEER_LEFT
                    - STEER_CENTER
                ),
                1.0,
            )
        )

        score = (
            DWA_W_GOAL_PROGRESS
            * goal_progress_score
            + DWA_W_GOAL_HEADING
            * goal_heading_score
            + DWA_W_CLEARANCE
            * clearance_score
            + DWA_W_FORWARD_OPENNESS
            * forward_openness_score
            - DWA_W_SEMANTIC_RISK
            * semantic_risk
            - DWA_W_STEERING
            * steering_penalty
        )

        candidate = {
            "steering_pwm": int(
                round(
                    float(
                        steering_pwm
                    )
                )
            ),
            "steering_angle_deg": math.degrees(
                steering_angle_rad
            ),
            "speed_mps": DWA_FORWARD_SPEED_MPS,
            "throttle_pwm": DWA_FORWARD_PWM,
            "trajectory_x": trajectory_x,
            "trajectory_y": trajectory_y,
            "collision": collision,
            "minimum_clearance_m": minimum_clearance,
            "forward_open_distance_m": forward_open_distance,
            "semantic_risk": semantic_risk,
            "score": score,
        }

        candidates.append(
            candidate
        )

        if not collision:
            safe_candidates.append(
                candidate
            )

    if not safe_candidates:
        return {
            "status": "STOP",
            "planner_lidar_points": int(len(lidar_points)),
            "steering_pwm": STEER_CENTER,
            "steering_angle_deg": 0.0,
            "speed_mps": 0.0,
            "throttle_pwm": DWA_STOP_PWM,
            "trajectory_x": np.array(
                [0.0]
            ),
            "trajectory_y": np.array(
                [0.0]
            ),
            "collision": False,
            "minimum_clearance_m": 0.0,
            "forward_open_distance_m": 0.0,
            "semantic_risk": 0.0,
            "score": 0.0,
            "safe_count": 0,
            "all_candidates": candidates,
        }

    best = max(
        safe_candidates,
        key=lambda item: item["score"],
    )

    result = dict(best)
    result["status"] = "FORWARD"
    result["planner_lidar_points"] = int(len(lidar_points))
    result["safe_count"] = len(
        safe_candidates
    )
    result["all_candidates"] = candidates

    return result


# ============================================================================
# LIDAR DRAWING
# ============================================================================

def draw_lidar_view(fused_objects=None, semantic_obstacles=None, dwa_result=None) -> tuple[np.ndarray, int]:
    canvas = np.zeros(
        (
            LIDAR_VIEW_SIZE,
            LIDAR_VIEW_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    center_x = LIDAR_VIEW_SIZE // 2
    center_y = int(
        LIDAR_VIEW_SIZE * 0.78
    )

    pixels_per_meter = (
        LIDAR_VIEW_SIZE
        * 0.42
        / LIDAR_MAX_DISPLAY_M
    )

    # Range arcs.
    for radius_m in (1, 2, 3, 4):
        radius_px = int(
            radius_m
            * pixels_per_meter
        )

        cv2.circle(
            canvas,
            (center_x, center_y),
            radius_px,
            (65, 65, 65),
            1,
        )

        cv2.putText(
            canvas,
            f"{radius_m}m",
            (
                center_x + 5,
                center_y - radius_px + 16,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 140),
            1,
        )

    # Forward line.
    cv2.line(
        canvas,
        (center_x, center_y),
        (
            center_x,
            max(
                0,
                center_y
                - int(
                    LIDAR_MAX_DISPLAY_M
                    * pixels_per_meter
                ),
            ),
        ),
        (90, 90, 90),
        1,
    )

    # Car footprint, roughly 13.25 in wide.
    car_width_px = max(
        8,
        int(
            0.3366
            * pixels_per_meter
        ),
    )

    car_length_px = max(
        14,
        int(
            0.56
            * pixels_per_meter
        ),
    )

    cv2.rectangle(
        canvas,
        (
            center_x - car_width_px // 2,
            center_y - car_length_px,
        ),
        (
            center_x + car_width_px // 2,
            center_y,
        ),
        (255, 255, 255),
        2,
    )

    with lidar_lock:
        distances = (
            latest_lidar_distances.copy()
        )
        scan_age = (
            time.time()
            - latest_lidar_time
            if latest_lidar_time > 0
            else 999.0
        )

    valid_count = 0
    collision_debug_points = (
        filter_lidar_collision_xy_from_bins(distances)
        if scan_age <= 1.0
        else np.empty((0, 2), dtype=float)
    )

    if scan_age <= 1.0:
        for index, distance_mm in enumerate(
            distances
        ):
            if distance_mm <= 0:
                continue

            distance_m = (
                float(distance_mm)
                / 1000.0
            )

            if (
                distance_m <= 0.02
                or distance_m
                > LIDAR_MAX_DISPLAY_M
            ):
                continue

            raw_angle_deg = (
                index
                * LIDAR_BIN_DEG
            )

            relative_angle_deg = (
                raw_angle_deg
                - LIDAR_FORWARD_OFFSET_DEG
            )

            theta = math.radians(
                relative_angle_deg
            )

            # 0 deg = screen forward/up.
            x_m = (
                distance_m
                * math.sin(theta)
            )

            y_m = (
                distance_m
                * math.cos(theta)
            )

            px = int(
                center_x
                + x_m
                * pixels_per_meter
            )

            py = int(
                center_y
                - y_m
                * pixels_per_meter
            )

            if (
                0 <= px < LIDAR_VIEW_SIZE
                and 0 <= py < LIDAR_VIEW_SIZE
            ):
                cv2.circle(
                    canvas,
                    (px, py),
                    4,
                    (255, 255, 255),
                    -1,
                )

                valid_count += 1

    # ------------------------------------------------------------------------
    # Live semantic obstacle field from real YOLO + LiDAR detections.
    # ------------------------------------------------------------------------

    if semantic_obstacles:
        for obstacle in semantic_obstacles:
            x_m = float(
                obstacle["x_m"]
            )
            y_m = float(
                obstacle["y_m"]
            )

            px = int(
                center_x
                + x_m
                * pixels_per_meter
            )

            py = int(
                center_y
                - y_m
                * pixels_per_meter
            )

            influence_m = min(
                float(
                    obstacle["influence_m"]
                ),
                SEMANTIC_FIELD_MAX_RADIUS_M,
            )

            risk = float(
                obstacle["risk"]
            )

            if risk >= SEMANTIC_FIELD_DRAW_MIN_RISK:
                radius_px = max(
                    8,
                    int(
                        influence_m
                        * pixels_per_meter
                    ),
                )

                # Higher-risk classes get a brighter red ring.
                intensity = int(
                    max(
                        60,
                        min(
                            255,
                            round(
                                255.0
                                * risk
                            ),
                        ),
                    )
                )

                cv2.circle(
                    canvas,
                    (
                        px,
                        py,
                    ),
                    radius_px,
                    (
                        0,
                        0,
                        intensity,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    canvas,
                    (
                        px,
                        py,
                    ),
                    max(
                        4,
                        int(
                            radius_px
                            * 0.55
                        ),
                    ),
                    (
                        0,
                        0,
                        min(
                            255,
                            intensity + 30,
                        ),
                    ),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    (
                        f"{obstacle['label']} "
                        f"({x_m:+.2f},{y_m:.2f})m"
                    ),
                    (
                        min(
                            px + 10,
                            LIDAR_VIEW_SIZE - 250,
                        ),
                        max(
                            py - radius_px - 6,
                            20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (
                        200,
                        200,
                        255,
                    ),
                    1,
                    cv2.LINE_AA,
                )

    # ------------------------------------------------------------------------
    # Live semantic DWA shadow-mode trajectories.
    # ------------------------------------------------------------------------

    if dwa_result is not None:
        all_candidates = dwa_result.get(
            "all_candidates",
            [],
        )

        # Draw the candidate fan so we can always see what DWA is evaluating.
        # Safe candidates are gray. Rejected collision candidates are dark red.
        for candidate_index, candidate in enumerate(
            all_candidates
        ):
            if candidate_index % 2 != 0:
                continue

            candidate_collision = candidate.get(
                "collision",
                False,
            )

            trajectory_x = candidate[
                "trajectory_x"
            ]

            trajectory_y = candidate[
                "trajectory_y"
            ]

            polyline_points = []

            for x_m, y_m in zip(
                trajectory_x,
                trajectory_y,
            ):
                px = int(
                    center_x
                    + float(x_m)
                    * pixels_per_meter
                )

                py = int(
                    center_y
                    - float(y_m)
                    * pixels_per_meter
                )

                polyline_points.append(
                    [px, py]
                )

            if len(polyline_points) >= 2:
                candidate_color = (
                    (70, 70, 70)
                    if not candidate_collision
                    else (0, 0, 90)
                )

                cv2.polylines(
                    canvas,
                    [
                        np.asarray(
                            polyline_points,
                            dtype=np.int32,
                        )
                    ],
                    False,
                    candidate_color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw local straight-ahead objective.
        goal_y_px = int(
            center_y
            - DWA_LOCAL_GOAL_DISTANCE_M
            * pixels_per_meter
        )

        if 0 <= goal_y_px < LIDAR_VIEW_SIZE:
            cv2.drawMarker(
                canvas,
                (
                    center_x,
                    goal_y_px,
                ),
                (180, 180, 180),
                cv2.MARKER_STAR,
                16,
                1,
                cv2.LINE_AA,
            )

        # Selected trajectory.
        selected_x = dwa_result.get(
            "trajectory_x"
        )

        selected_y = dwa_result.get(
            "trajectory_y"
        )

        if (
            selected_x is not None
            and selected_y is not None
            and len(selected_x) >= 2
        ):
            selected_points = []

            for x_m, y_m in zip(
                selected_x,
                selected_y,
            ):
                px = int(
                    center_x
                    + float(x_m)
                    * pixels_per_meter
                )

                py = int(
                    center_y
                    - float(y_m)
                    * pixels_per_meter
                )

                selected_points.append(
                    [px, py]
                )

            cv2.polylines(
                canvas,
                [
                    np.asarray(
                        selected_points,
                        dtype=np.int32,
                    )
                ],
                False,
                (0, 255, 0),
                4,
                cv2.LINE_AA,
            )

            end_px, end_py = (
                selected_points[-1]
            )

            cv2.drawMarker(
                canvas,
                (
                    int(end_px),
                    int(end_py),
                ),
                (0, 255, 0),
                cv2.MARKER_TILTED_CROSS,
                12,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "SEMANTIC DWA PLANNER",
            (
                18,
                LIDAR_VIEW_SIZE - 48,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        proposed_status = str(
            dwa_result.get(
                "status",
                "UNKNOWN",
            )
        )

        cv2.putText(
            canvas,
            (
                f"Proposed: {proposed_status} | steer "
                f"{int(dwa_result.get('steering_pwm', STEER_CENTER))} "
                f"| throttle "
                f"{int(dwa_result.get('throttle_pwm', DWA_STOP_PWM))}"
            ),
            (
                18,
                LIDAR_VIEW_SIZE - 22,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Highlight fused semantic detections.
    if fused_objects:
        for fused_object in fused_objects:
            distance_m = fused_object.get("distance_m")
            bearing_deg = fused_object.get("lidar_bearing_deg")

            if (
                distance_m is None
                or bearing_deg is None
            ):
                continue

            theta = math.radians(float(bearing_deg))

            x_m = float(distance_m) * math.sin(theta)
            y_m = float(distance_m) * math.cos(theta)

            px = int(
                center_x + x_m * pixels_per_meter
            )
            py = int(
                center_y - y_m * pixels_per_meter
            )

            if (
                0 <= px < LIDAR_VIEW_SIZE
                and 0 <= py < LIDAR_VIEW_SIZE
            ):
                cv2.circle(
                    canvas,
                    (px, py),
                    13,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    (
                        f"{fused_object['label']} "
                        f"{float(distance_m):.2f}m "
                        f"{float(bearing_deg):+.0f}deg"
                    ),
                    (
                        min(px + 14, LIDAR_VIEW_SIZE - 230),
                        max(py - 12, 25),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.putText(
        canvas,
        "LIVE LD19 LIDAR",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        canvas,
        f"raw pts {valid_count} | collision pts {len(collision_debug_points)}",
        (12, LIDAR_VIEW_SIZE - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


    return canvas, valid_count


# ============================================================================
# CAMERA + YOLO ASYNC WORKERS
# ============================================================================

camera_lock = threading.Lock()
latest_camera_frame = None
latest_camera_frame_time = 0.0
camera_frame_sequence = 0
camera_running = True
camera_connected = False
camera_read_errors = 0

def camera_receiver():
    global latest_camera_frame, latest_camera_frame_time, camera_frame_sequence
    global camera_running, camera_connected, camera_read_errors
    cap_local = None
    last_print = 0.0
    while camera_running:
        if cap_local is None or not cap_local.isOpened():
            if cap_local is not None:
                cap_local.release()
            cap_local = cv2.VideoCapture(STREAM_URL)
            cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap_local.isOpened():
                camera_connected = False
                if time.time() - last_print >= 2.0:
                    print("Could not open ESP32 camera stream. Retrying...")
                    last_print = time.time()
                cap_local.release(); cap_local = None
                time.sleep(0.35); continue
        ok, frame = cap_local.read()
        if not ok or frame is None:
            camera_read_errors += 1; camera_connected = False
            cap_local.release(); cap_local = None
            time.sleep(0.08); continue
        with camera_lock:
            latest_camera_frame = frame
            latest_camera_frame_time = time.time()
            camera_frame_sequence += 1
        camera_connected = True
    if cap_local is not None:
        cap_local.release()

camera_thread = threading.Thread(target=camera_receiver, daemon=True)
camera_thread.start()

# ============================================================================
# YOLO
# ============================================================================

if not MODEL_PATH.is_file():
    print("YOLO V4.6 model not found:")
    print(MODEL_PATH.resolve())
    camera_running = False; lidar_running = False; control_thread_running = False
    control_sock.close(); raise SystemExit(1)

print(f"Loading YOLO V4.6: {MODEL_PATH.resolve()}")
model = YOLO(str(MODEL_PATH))
print("Classes:", model.names)
ensure_capture_log()

yolo_lock = threading.Lock()
latest_yolo_boxes = []
latest_yolo_time = 0.0
latest_yolo_inference_ms = 0.0
yolo_running = True

def yolo_receiver():
    global latest_yolo_boxes, latest_yolo_time, latest_yolo_inference_ms, yolo_running
    next_run = time.perf_counter()
    last_seq = -1
    while yolo_running:
        nowp = time.perf_counter()
        if nowp < next_run:
            time.sleep(min(next_run-nowp, 0.005)); continue
        next_run = nowp + YOLO_PERIOD_S
        with camera_lock:
            frame = None if latest_camera_frame is None else latest_camera_frame.copy()
            seq = camera_frame_sequence; ft = latest_camera_frame_time
        if frame is None or seq == last_seq or time.time()-ft > CAMERA_STALE_S:
            continue
        t0=time.perf_counter()
        try:
            results=model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
            boxes=[]
            for result in results:
                for box in result.boxes:
                    x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                    conf=float(box.conf[0]); ci=int(box.cls[0]); label=str(model.names[ci])
                    boxes.append((float(x1),float(y1),float(x2),float(y2),label,conf))
            with yolo_lock:
                latest_yolo_boxes=boxes
                latest_yolo_time=time.time()
                latest_yolo_inference_ms=(time.perf_counter()-t0)*1000.0
            last_seq=seq
        except Exception as e:
            print(f"YOLO error: {e}"); time.sleep(0.05)

yolo_thread=threading.Thread(target=yolo_receiver, daemon=True)
yolo_thread.start()

# ============================================================================
# RUNTIME STATE
# ============================================================================

armed = False
control_mode = "MANUAL"
auto_mode_enter_time = 0.0

previous_frame_time = time.time()

inference_ms = 0.0
cached_boxes = []

status_message = "DISARMED"
status_message_time = time.time()
last_semantic_print_time = 0.0


def set_status(message: str) -> None:
    global status_message
    global status_message_time

    status_message = message
    status_message_time = time.time()

    print(message)


# Put the RC outputs in a known-safe state immediately.
emergency_stop()

print()
print("=" * 72)
print("  B = FULL AUTO Semantic-DWA (steering + throttle)")
print("YOLO V4.6 + LD19 LIVE DASHBOARD")
print("=" * 72)
print("Connect laptop to RC_CAR_WIFI.")
print("R arm | T disarm | SPACE emergency stop")
print("M manual mode | V DWA steering-assist mode")
print("Manual: W/S throttle + A/D steering")
print("Assist: hold W for forward throttle; DWA controls steering")
print(
    f"Independent control heartbeat: {CONTROL_TX_HZ:.0f} Hz "
    f"({CONTROL_TX_PERIOD_S * 1000.0:.0f} ms target interval)"
)
print("P missed | L wrong label | F false detection | G general")
print("Q quit")
print("MULTI-CLASS SEMANTIC FUSION BUILD ACTIVE")
print("LIVE SEMANTIC DWA ACTIVE")
print("V4.5: dense-close hazard now requires 7 contiguous, range-consistent bins")
print("V4.4: DWA clearance tuned for physical car: collision radius 0.23 m, full score 0.80 m")
print("V4.3: grass near-field suppression active; semantic fusion min range 0.45 m")
print("STARTING IN MANUAL MODE")
print("DWA gains steering authority only after pressing V")
print(
    f"Camera HFOV estimate: {CAMERA_HFOV_DEG:.1f} deg"
)
print(f"Camera stream: {STREAM_URL}")
print(f"LiDAR scan: {LIDAR_SCAN_URL}")
print(f"LiDAR status: {LIDAR_STATUS_URL}")
print(f"LiDAR HTTP poll target: {LIDAR_POLL_HZ:.0f} Hz")
print(f"RC control UDP: {ESP32_IP}:{CONTROL_UDP_PORT} (CTRL:steer,throttle)")
print("=" * 72)


# ============================================================================
# MAIN LOOP
# ============================================================================

try:
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    running = True

    while running:
        with camera_lock:
            raw_frame = None if latest_camera_frame is None else latest_camera_frame.copy()
            camera_time_snapshot = latest_camera_frame_time
        if raw_frame is None:
            time.sleep(0.01)
            if pressed_once("q"):
                running=False; break
            continue
        camera_age = time.time()-camera_time_snapshot if camera_time_snapshot > 0 else 999.0

        now = time.time()

        raw_h, raw_w = (
            raw_frame.shape[:2]
        )

        # --------------------------------------------------------------------
        # Keys
        # --------------------------------------------------------------------

        if pressed_once("q"):
            running = False
            break

        if pressed_once("space"):
            armed = False
            control_mode = "MANUAL"
            emergency_stop()
            set_status(
                "EMERGENCY STOP - MANUAL MODE"
            )

        if pressed_once("r"):
            armed = True
            set_status(
                f"ARMED - {control_mode}"
            )

        if pressed_once("t"):
            armed = False
            emergency_stop()
            set_status("DISARMED")

        if pressed_once("m"):
            control_mode = "MANUAL"
            set_desired_command(
                STEER_CENTER,
                ESC_NEUTRAL,
                armed,
            )
            set_status("MANUAL MODE")

        if pressed_once("v"):
            control_mode = "DWA_ASSIST"
            set_desired_command(
                STEER_CENTER,
                ESC_NEUTRAL,
                armed,
            )
            set_status(
                "DWA STEERING ASSIST - HOLD W TO MOVE"
            )

        if pressed_once("b"):
            control_mode = "DWA_AUTO"
            auto_mode_enter_time = time.time()
            set_desired_command(
                STEER_CENTER,
                ESC_NEUTRAL,
                armed,
            )
            set_status(
                "FULL AUTO SELECTED - 2 SECOND READY DELAY"
            )

        # --------------------------------------------------------------------
        # Read user input. Final command arbitration happens after DWA runs.
        # --------------------------------------------------------------------

        manual_steering_us = STEER_CENTER
        requested_throttle_us = ESC_NEUTRAL

        if keyboard.is_pressed("a"):
            manual_steering_us = STEER_LEFT
        elif keyboard.is_pressed("d"):
            manual_steering_us = STEER_RIGHT

        if keyboard.is_pressed("w"):
            requested_throttle_us = ESC_FORWARD
        elif keyboard.is_pressed("s"):
            requested_throttle_us = ESC_REVERSE

        steering_us = manual_steering_us
        throttle_us = requested_throttle_us

        # --------------------------------------------------------------------
        # YOLO snapshot from asynchronous inference worker
        # --------------------------------------------------------------------
        with yolo_lock:
            yolo_age = time.time()-latest_yolo_time if latest_yolo_time > 0 else 999.0
            cached_boxes = list(latest_yolo_boxes) if yolo_age <= YOLO_MAX_RESULT_AGE_S and camera_age <= CAMERA_STALE_S else []
            inference_ms = float(latest_yolo_inference_ms)

        # --------------------------------------------------------------------
        # Capture
        # --------------------------------------------------------------------

        capture_category = None

        if pressed_once("p"):
            capture_category = "missed"
        elif pressed_once("l"):
            capture_category = "wrong_label"
        elif pressed_once("f"):
            capture_category = "false_detection"
        elif pressed_once("g"):
            capture_category = "general"

        if capture_category:
            save_training_frame(
                raw_frame.copy(),
                capture_category,
                armed,
                steering_us,
                throttle_us,
                cached_boxes,
            )

            set_status(
                f"SAVED: {capture_category}"
            )

        # --------------------------------------------------------------------
        # YOLO + LiDAR semantic fusion
        # --------------------------------------------------------------------

        cached_boxes = filter_yolo_boxes(
            cached_boxes,
            raw_w,
            raw_h,
        )

        fused_objects = fuse_yolo_detections(
            cached_boxes,
            raw_w,
        )

        semantic_obstacles = build_semantic_obstacle_list(
            fused_objects
        )


        if (
            semantic_obstacles
            and now - last_semantic_print_time >= 1.0
        ):
            last_semantic_print_time = now

            summary = " | ".join(
                (
                    f"{item['label']} "
                    f"x={float(item['x_m']):+.2f}m "
                    f"y={float(item['y_m']):.2f}m "
                    f"conf={float(item['confidence']):.2f}"
                )
                for item in semantic_obstacles
            )

            print(
                "SEMANTIC OBSTACLES: "
                + summary
            )

        # --------------------------------------------------------------------
        # LIVE SEMANTIC DWA SHADOW MODE
        # --------------------------------------------------------------------
        #
        # This computes a recommendation only.
        # DWA output is NOT passed to send_command().
        #
        dwa_result = run_semantic_dwa_shadow(
            semantic_obstacles
        )

        # --------------------------------------------------------------------
        # CONTROL AUTHORITY + HARD LIDAR SAFETY
        # --------------------------------------------------------------------

        with lidar_lock:
            assist_lidar_age = (
                time.time() - latest_lidar_time
                if latest_lidar_time > 0
                else 999.0
            )

        lidar_fresh_for_assist = (
            assist_lidar_age
            <= ASSIST_LIDAR_MAX_AGE_S
        )

        assist_block_reason = ""

        if control_mode == "DWA_ASSIST":
            if dwa_result["status"] == "FORWARD":
                steering_us = int(
                    dwa_result["steering_pwm"]
                )
            else:
                steering_us = STEER_CENTER

            # Current DWA predicts forward motion only. Reverse stays manual.
            if requested_throttle_us == ESC_REVERSE:
                throttle_us = ESC_NEUTRAL
                assist_block_reason = (
                    "ASSIST REVERSE DISABLED - PRESS M FOR MANUAL"
                )

            elif requested_throttle_us == ESC_FORWARD:
                if not lidar_fresh_for_assist:
                    throttle_us = ESC_NEUTRAL
                    steering_us = STEER_CENTER
                    assist_block_reason = (
                        "BLOCKED: LIDAR DATA STALE/LOST"
                    )

                elif (
                    ASSIST_REQUIRE_SAFE_TRAJECTORY
                    and dwa_result["safe_count"] <= 0
                ):
                    throttle_us = ESC_NEUTRAL
                    steering_us = STEER_CENTER
                    assist_block_reason = (
                        "BLOCKED: NO SAFE LIDAR TRAJECTORY"
                    )

                elif dwa_result["status"] != "FORWARD":
                    throttle_us = ESC_NEUTRAL
                    steering_us = STEER_CENTER
                    assist_block_reason = (
                        "BLOCKED: DWA REQUESTED STOP"
                    )

                else:
                    throttle_us = ESC_FORWARD

            else:
                # Dead-man behavior: release W -> immediate neutral.
                throttle_us = ESC_NEUTRAL

        elif control_mode == "DWA_AUTO":
            # DWA owns both steering and throttle in this mode.
            # The same LiDAR freshness and planner-validity gates used by
            # steering assist remain mandatory.
            auto_ready = (
                time.time() - auto_mode_enter_time
                >= AUTO_ENTRY_DELAY_S
            )

            if not auto_ready:
                steering_us = STEER_CENTER
                throttle_us = ESC_NEUTRAL
                assist_block_reason = "AUTO READY DELAY"

            elif not lidar_fresh_for_assist:
                steering_us = STEER_CENTER
                throttle_us = ESC_NEUTRAL
                assist_block_reason = "AUTO BLOCKED - LIDAR STALE"

            elif dwa_result["status"] != "FORWARD":
                steering_us = STEER_CENTER
                throttle_us = ESC_NEUTRAL
                assist_block_reason = "AUTO BLOCKED - NO SAFE PATH"

            else:
                steering_us = int(
                    dwa_result["steering_pwm"]
                )
                throttle_us = AUTO_FORWARD_PWM
                assist_block_reason = "FULL SEMANTIC-DWA AUTO"

        else:
            # Manual mode is the intentional user override.
            steering_us = manual_steering_us
            throttle_us = requested_throttle_us

        if not armed:
            throttle_us = ESC_NEUTRAL

        set_desired_command(
            steering_us,
            throttle_us,
            armed,
        )

        # --------------------------------------------------------------------
        # Camera display
        # --------------------------------------------------------------------

        camera_display = cv2.resize(
            raw_frame,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        scale_x = (
            CAMERA_WIDTH / raw_w
        )

        scale_y = (
            CAMERA_HEIGHT / raw_h
        )

        for (
            x1,
            y1,
            x2,
            y2,
            label,
            confidence,
        ) in cached_boxes:

            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)

            cv2.rectangle(
                camera_display,
                (sx1, sy1),
                (sx2, sy2),
                (255, 255, 255),
                2,
            )

            display_label = (
                f"{label} {confidence:.2f}"
            )

            box_center_x = (
                float(x1) + float(x2)
            ) / 2.0

            matching_object = None
            smallest_error = float("inf")

            for fused_object in fused_objects:
                if fused_object["label"] != label:
                    continue

                fx1, fy1, fx2, fy2 = (
                    fused_object["bbox"]
                )

                fused_center_x = (
                    float(fx1)
                    + float(fx2)
                ) / 2.0

                center_error = abs(
                    fused_center_x
                    - box_center_x
                )

                if center_error < smallest_error:
                    smallest_error = center_error
                    matching_object = fused_object

            if matching_object is not None:
                bearing = matching_object[
                    "camera_bearing_deg"
                ]

                distance_m = matching_object[
                    "distance_m"
                ]

                if distance_m is not None:
                    display_label = (
                        f"{label} {confidence:.2f} | "
                        f"{float(distance_m):.2f}m | "
                        f"{float(bearing):+.0f}deg"
                    )
                else:
                    display_label = (
                        f"{label} {confidence:.2f} | "
                        f"{float(bearing):+.0f}deg | "
                        "NO LIDAR MATCH"
                    )

            cv2.putText(
                camera_display,
                display_label,
                (
                    sx1,
                    max(
                        sy1 - 8,
                        20,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            camera_display,
            "CAMERA + YOLO V4.6",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

        # --------------------------------------------------------------------
        # LiDAR view
        # --------------------------------------------------------------------

        lidar_display, valid_points = (
            draw_lidar_view(
                fused_objects,
                semantic_obstacles,
                dwa_result,
            )
        )

        with lidar_lock:
            lidar_age = (
                time.time()
                - latest_lidar_time
                if latest_lidar_time > 0
                else 999.0
            )

            lidar_sequence = (
                latest_lidar_sequence
            )

            lidar_rate = (
                latest_lidar_scan_rate_hz
            )

        # --------------------------------------------------------------------
        # Compact status bar
        # --------------------------------------------------------------------

        fps = 1.0 / max(
            now - previous_frame_time,
            0.001,
        )
        previous_frame_time = now

        with lidar_lock:
            lidar_scans_received = latest_lidar_http_scans
            lidar_http_errors = latest_lidar_http_errors

        status_bar = np.zeros(
            (
                STATUS_HEIGHT,
                WINDOW_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        lidar_state = (
            "LIVE"
            if lidar_age <= 0.5
            else "NO DATA"
        )

        tx_rate_display = float(
            control_tx_rate_hz
        )
        tx_last_gap_ms = float(
            control_tx_last_gap_s * 1000.0
        )
        tx_max_gap_ms = float(
            control_tx_max_gap_s * 1000.0
        )

        line1 = (
            f"{'ARMED' if armed else 'DISARMED'}"
            f"   |   MODE {control_mode}"
            f"   |   CMD {steering_us}/{throttle_us}"
            f"   |   TX {tx_rate_display:.1f} Hz"
            f" gap {tx_last_gap_ms:.0f} ms"
            f" max {tx_max_gap_ms:.0f} ms"
            f"   |   UI {fps:.1f} FPS"
            f"   |   Cam age {camera_age * 1000.0:.0f} ms"
            f"   |   YOLO {inference_ms:.0f} ms"
        )

        fused_match_count = sum(
            1
            for fused_object in fused_objects
            if fused_object["distance_m"] is not None
        )


        nearest_semantic = None

        if semantic_obstacles:
            nearest_semantic = min(
                semantic_obstacles,
                key=lambda item:
                float(
                    item["distance_m"]
                ),
            )

        control_gap_warning = (
            control_tx_last_gap_s >= 0.40
        )

        if control_gap_warning:
            line2 = (
                "CONTROL WARNING: "
                f"TX GAP {control_tx_last_gap_s * 1000.0:.0f} ms "
                "(ESP32 failsafe is 500 ms)"
            )
        elif assist_block_reason:
            line2 = (
                f"SAFETY: {assist_block_reason}"
            )
        else:
            line2 = (
                f"LiDAR {lidar_state}"
                f"   |   Scan {lidar_rate:.2f} Hz"
                f" HTTP={lidar_scans_received}"
                f" err={lidar_http_errors}"
                f"   |   Fusion "
                f"{fused_match_count}/{len(fused_objects)}"
                f"   |   Semantic {len(semantic_obstacles)}"
                f"   |   DWA {dwa_result['status']}"
                f" closeHaz={int(has_dense_close_hazard(latest_lidar_distances))}"
                f" safe={int(dwa_result['safe_count'])}"
                f" pts={int(dwa_result.get('planner_lidar_points', 0))}"
                f"   |   Open "
                f"{float(dwa_result['forward_open_distance_m']):.2f}m"
            )

        cv2.putText(
            status_bar,
            line1,
            (18, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            status_bar,
            line2,
            (18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        top = np.hstack(
            (
                camera_display,
                lidar_display,
            )
        )

        dashboard = np.vstack(
            (
                top,
                status_bar,
            )
        )

        cv2.imshow(
            WINDOW_NAME,
            dashboard,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == 27
        ):
            running = False

finally:
    armed = False

    emergency_stop()

    set_desired_command(
        STEER_CENTER,
        ESC_NEUTRAL,
        False,
    )

    time.sleep(0.10)

    control_thread_running = False
    lidar_running = False
    camera_running = False
    yolo_running = False

    try:
        control_sock.close()
    except OSError:
        pass

    cv2.destroyAllWindows()

    print()
    print("Dashboard closed safely.")