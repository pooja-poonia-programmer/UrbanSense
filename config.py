from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
for d in (MODELS_DIR, ASSETS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

YOLO_MODEL = MODELS_DIR / "yolo26n.pt"
POTHOLE_MODEL = MODELS_DIR / "pothole_model.pt"
SIGN_MODEL = MODELS_DIR / "traffic_sign_model.pt"
PLATE_MODEL = MODELS_DIR / "license_plate_model.pt"
DEMO_VIDEO = ASSETS_DIR / "road_video.mp4"

# Public sources used only when a local weight is missing.
POTHOLE_URL = "https://github.com/Yug-doshi/PotholeGuard-AI/raw/refs/heads/main/yolov8_pothole_best.pt"
SIGN_HF_REPO = "ankitjha07/Traffic-Sign-Detection-YOLOv8"
PLATE_HF_REPO = "Babblu2821/alpr-plate-detector"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

YOLO_CONF = 0.25
POTHOLE_CONF = 0.40
SIGN_CONF = 0.40
PLATE_CONF = 0.35
OCR_CONF = 0.45

SPECIALIST_EVERY = 5
TARGET_MAX_DIM = 1280
MAX_FILE_FRAMES = 6000
DEFAULT_LIVE_SECONDS = 20
PROXIMITY_RATIO = 0.035
MOTION_RATIO = 0.035
EVENT_COOLDOWN_SECONDS = 3.0

COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}

DEMO_BUS_ID = "BUS-DEMO-01"
DEFAULT_ROUTE_START = {"lat": 30.7046, "lon": 76.7179}
DEFAULT_ROUTE_END = {"lat": 30.7146, "lon": 76.7279}
DEFAULT_FIXED_GPS = {"lat": 30.7046, "lon": 76.7179}

DEMO_TAG = "DEMO EXAMPLE"
REAL_TAG = "VIDEO DERIVED"
CANDIDATE_TAG = "CANDIDATE"
SIM_TAG = "SIMULATED PRESENTATION SCENARIO"

ACTIONS = {
    "POTHOLE_DETECTED": "Prioritize road inspection and maintenance dispatch.",
    "ROAD_DEFECT_CANDIDATE": "Schedule scene verification by road-maintenance staff.",
    "WATERLOGGING_CANDIDATE": "Alert drainage/municipal response team for inspection.",
    "TRAFFIC_SIGN_CANDIDATE": "Inspect signboard condition and visibility.",
    "TRAFFIC_SIGNAL_CANDIDATE": "Inspect signal state/location and controller health.",
    "PEDESTRIAN_PROXIMITY": "Prioritize vulnerable-road-user safety inspection.",
    "ABNORMAL_MOTION_CANDIDATE": "Send candidate to traffic-enforcement review.",
    "HIT_AND_RUN_CANDIDATE": "Escalate evidence to traffic police for manual review.",
    "PLATE_DETECTED": "Associate plate evidence with an incident only after verification.",
    "CONGESTION_HOTSPOT": "Review traffic-flow and signal-timing options.",
}
