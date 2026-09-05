import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from ultralytics import YOLO

def calculate_pedestrian_risk(people, vehicles):
    # Simple geometric proximity rule used by the proven local prototype.
    # It is a review flag, not a collision prediction.
    for px1, py1, px2, py2 in people:
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        for vx1, vy1, vx2, vy2 in vehicles:
            vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
            vdiag = max(1.0, math.hypot(vx2 - vx1, vy2 - vy1))
            if math.hypot(pcx - vcx, pcy - vcy) / vdiag < 2.0:
                return {"risk": True}
    return {"risk": False}


st.set_page_config(page_title="UrbanSense", page_icon="🚌", layout="wide")

def model_path(filename):
    candidates = [filename, os.path.join("models", filename)]
    for path in candidates:
        if os.path.exists(path):
            return path
    return filename

@st.cache_resource
def load_vehicle_model():
    return YOLO(model_path("yolo26n.pt"))

@st.cache_resource
def load_pothole_model():
    return YOLO(model_path("pothole_model.pt")) if os.path.exists(model_path("pothole_model.pt")) else None

@st.cache_resource
def load_sign_model():
    return YOLO(model_path("traffic_sign_model.pt")) if os.path.exists(model_path("traffic_sign_model.pt")) else None

@st.cache_resource
def load_plate_model():
    return YOLO(model_path("license_plate_model.pt")) if os.path.exists(model_path("license_plate_model.pt")) else None

@st.cache_resource
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None

vehicle_model = load_vehicle_model()
pothole_model = load_pothole_model()
sign_model = load_sign_model()
plate_model = load_plate_model()
ocr_reader = load_ocr() if plate_model else None

if "result" not in st.session_state:
    st.session_state.result = None


def event(event_type, confidence, frame, fps, details=None):
    return {
        "event_id": "EVT-" + datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "event_type": event_type,
        "confidence": round(float(confidence), 3),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_time_s": round(frame / fps, 2),
        "details": details or {},
    }


def severity(conf):
    if conf >= .80: return "HIGH"
    if conf >= .55: return "MEDIUM"
    return "REVIEW"


def water_candidate(frame):
    """Visual water-like surface candidate; not a flood-depth measurement."""
    h, w = frame.shape[:2]
    roi = frame[int(h * .55):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Detect broad blue/cyan or dark reflective areas in the road region.
    blue = cv2.inRange(hsv, np.array([85, 35, 25]), np.array([135, 255, 230]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 90, 95]))
    ratio = max(cv2.countNonZero(blue), cv2.countNonZero(dark)) / float(roi.shape[0] * roi.shape[1])
    return min(ratio * 2.0, 1.0) if ratio > .08 else 0.0


def road_marking_candidate(frame):
    """Find lane/zebra-like bright markings; cannot prove a marking is missing."""
    h, w = frame.shape[:2]
    roi = frame[int(h * .55):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 180, 255)
    ratio = cv2.countNonZero(bright) / float(bright.size)
    return min(ratio * 3.0, 1.0)


def run_analysis(video_path, latitude=None, longitude=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the selected video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release(); raise RuntimeError("Invalid video dimensions.")

    raw = "urbansense_demo_raw.mp4"
    web = "urbansense_demo_web.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = []
    class_totals = {"car":0, "bus":0, "truck":0, "motorcycle":0}
    events = []
    max_people = 0
    max_nearby = 0
    risk_streak = 0
    max_vehicles = 0
    pothole_frames = 0
    pothole_conf = 0.0
    signs = 0
    plates = 0
    ocr_candidates = []
    water_hits = 0
    marking_hits = 0
    track_history = {}
    motion_streak = {}
    motion_events = 0
    frame_no = 0
    last_pothole_event_frame = -9999
    last_water_event_frame = -9999

    progress = st.progress(0)
    status = st.empty()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_no += 1

        result = vehicle_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        people, vehicles = [], []
        ids = result.boxes.id.int().cpu().tolist() if result.boxes is not None and result.boxes.id is not None else []
        boxes = result.boxes if result.boxes is not None else []

        for i, box in enumerate(boxes):
            cid = int(box.cls[0]); name = vehicle_model.names[cid]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            if name == "person": people.append((x1,y1,x2,y2))
            if name in class_totals:
                vehicles.append((x1,y1,x2,y2)); class_totals[name] += 1
                tid = ids[i] if i < len(ids) else None
                if tid is not None:
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    diag = max(1.0, math.hypot(x2-x1, y2-y1))
                    if tid in track_history:
                        motion = math.hypot(cx-track_history[tid][0], cy-track_history[tid][1]) / diag
                        motion_streak[tid] = motion_streak.get(tid,0)+1 if motion > .35 else 0
                        if motion_streak[tid] == 3:
                            motion_events += 1
                            e = event("ABNORMAL_VEHICLE_MOTION_CANDIDATE", min(.99,.5+motion/2), frame_no, fps, {"track_id":int(tid),"vehicle_type":name,"normalized_motion":round(motion,3),"review_required":True})
                            if latitude is not None: e.update(latitude=latitude, longitude=longitude)
                            events.append(e)
                    track_history[tid] = (cx,cy)

        vc = len(vehicles); counts.append(vc); max_vehicles = max(max_vehicles, vc)
        risk = calculate_pedestrian_risk(people, vehicles)
        risk_streak = risk_streak + 1 if risk["risk"] else 0
        max_people = max(max_people, len(people)); max_nearby = max(max_nearby, len(vehicles))
        if risk_streak == 3:
            e = event("PEDESTRIAN_VEHICLE_PROXIMITY_RISK", .75, frame_no, fps, {"people":len(people),"vehicles":len(vehicles),"review_required":True})
            if latitude is not None: e.update(latitude=latitude, longitude=longitude)
            events.append(e)

        # Dedicated road model runs periodically on actual camera frames.
        if frame_no % 5 == 0:
            pr = pothole_model(frame, conf=.25, verbose=False)[0] if pothole_model is not None else None
            n = len(pr.boxes) if pr is not None and pr.boxes is not None else 0
            if n:
                pothole_frames += n
                cs = pr.boxes.conf.tolist(); strongest = max(cs) if cs else 0
                pothole_conf = max(pothole_conf, strongest)
                if frame_no - last_pothole_event_frame > int(fps*2):
                    e = event("POTHOLE_DETECTED", strongest, frame_no, fps, {"count_in_frame":n,"severity":severity(strongest)})
                    if latitude is not None: e.update(latitude=latitude, longitude=longitude)
                    events.append(e); last_pothole_event_frame = frame_no

            wc = water_candidate(frame)
            if wc > .25:
                water_hits += 1
                if frame_no - last_water_event_frame > int(fps*3):
                    e = event("WATERLOGGING_VISUAL_CANDIDATE", wc, frame_no, fps, {"method":"visual road-surface heuristic","review_required":True})
                    if latitude is not None: e.update(latitude=latitude, longitude=longitude)
                    events.append(e); last_water_event_frame = frame_no

            if road_marking_candidate(frame) > .25:
                marking_hits += 1

            if sign_model is not None:
                sr = sign_model(frame, conf=.25, verbose=False)[0]
                signs += len(sr.boxes) if sr.boxes is not None else 0

            if plate_model is not None:
                lr = plate_model(frame, conf=.20, verbose=False)[0]
                if lr.boxes is not None and len(lr.boxes):
                    plates += len(lr.boxes)
                    if ocr_reader is not None:
                        best = max(lr.boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1])))
                        x1,y1,x2,y2 = map(int,best.xyxy[0])
                        crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                        if crop.size and crop.shape[1] >= 60 and crop.shape[0] >= 20:
                            try:
                                for item in ocr_reader.readtext(crop, detail=1, paragraph=False):
                                    if item[1].strip() and float(item[2]) >= .40:
                                        ocr_candidates.append({"text":item[1].strip(),"confidence":round(float(item[2]),3),"frame":frame_no})
                            except Exception: pass

        annotated = result.plot()
        texts = [f"UrbanSense | Frame {frame_no}", f"Vehicles: {vc}", f"People: {len(people)}"]
        if risk["risk"]: texts.append("PEDESTRIAN SAFETY: REVIEW")
        if frame_no % 5 == 0 and n: texts.append(f"POTHOLES: {n}")
        y=32
        for text in texts:
            cv2.putText(annotated,text,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA); y += 28
        writer.write(annotated)
        if total: progress.progress(min(frame_no/total,1.0))
        if frame_no % 10 == 0: status.write(f"Processing camera stream: {frame_no}/{total or '?'} frames")

    cap.release(); writer.release(); progress.empty(); status.empty()

    density = "LOW" if max_vehicles <= 2 else "MEDIUM" if max_vehicles <= 5 else "HIGH"
    avg = round(sum(counts)/len(counts),2) if counts else 0
    # Browser conversion using installed imageio-ffmpeg.
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        rc = os.system(f'"{ffmpeg}" -y -i "{raw}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "{web}"')
    except Exception:
        rc = 1
    output = web if rc == 0 and os.path.exists(web) else raw
    return {"events":events,"output":output,"max_vehicles":max_vehicles,"average":avg,"density":density,"class_totals":class_totals,"max_people":max_people,"max_nearby":max_nearby,"ped_risk":risk_streak>=3,"potholes":pothole_frames,"pothole_conf":pothole_conf,"signs":signs,"plates":plates,"ocr":ocr_candidates,"water_hits":water_hits,"marking_hits":marking_hits,"motion_events":motion_events,"gps":(latitude,longitude)}


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🚌 UrbanSense")
st.subheader("AI-Powered Mobile Urban Intelligence Platform")
st.success("🟢 SYSTEM ONLINE")
st.write("Turning public-transport camera footage into actionable urban intelligence.")

st.markdown("### 🎥 Camera / CCTV Input")
mode = st.radio("Video source", ["Use demo video", "Upload camera video"], horizontal=True)
video = None
if mode == "Use demo video":
    demo_candidates = ["road_video.mp4", os.path.join("assets", "road_video.mp4")]
    demo_video = next((p for p in demo_candidates if os.path.exists(p)), None)
    if demo_video:
        video = demo_video
        st.video(video)
    else:
        st.error("road_video.mp4 is not present.")
else:
    up = st.file_uploader("Upload road / bus-camera video", type=["mp4","mov","avi","mkv"])
    if up:
        video = "uploaded_camera_input.mp4"
        with open(video,"wb") as f: f.write(up.getbuffer())
        st.video(up)

with st.expander("📍 Camera GPS metadata (optional)"):
    st.caption("UrbanSense never invents GPS coordinates. Supply GPS only when it is actually available from the camera/bus system.")
    use_gps = st.checkbox("Attach supplied GPS to generated events")
    lat = lon = None
    if use_gps:
        a,b=st.columns(2)
        lat=a.number_input("Latitude", value=0.0, format="%.6f")
        lon=b.number_input("Longitude", value=0.0, format="%.6f")

if st.button("▶ Run Full AI Analysis", type="primary", disabled=video is None):
    with st.spinner("Running UrbanSense edge-AI pipeline..."):
        started=time.time()
        try:
            st.session_state.result=run_analysis(video, lat if use_gps else None, lon if use_gps else None)
            st.session_state.result["seconds"]=round(time.time()-started,1)
            st.success(f"✅ Analysis complete in {st.session_state.result['seconds']} seconds.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

res=st.session_state.result
if res:
    st.markdown("---")
    st.markdown("## 🏙️ City Command Center")
    a,b,c,d=st.columns(4)
    a.metric("Peak Vehicles",res["max_vehicles"]); b.metric("Traffic Density",res["density"]); c.metric("Road Defect Detections",res["potholes"]); d.metric("AI Events",len(res["events"]))

    st.markdown("### 🚦 Traffic Intelligence")
    a,b,c,d=st.columns(4)
    a.metric("Cars",res["class_totals"]["car"]); b.metric("Buses",res["class_totals"]["bus"]); c.metric("Trucks",res["class_totals"]["truck"]); d.metric("Motorcycles",res["class_totals"]["motorcycle"])
    st.write(f"Average vehicles per processed frame: **{res['average']}**")

    st.markdown("### 🕳️ Road Health Intelligence")
    score=max(0,100-min(res["potholes"],4)*15-{"LOW":0,"MEDIUM":8,"HIGH":15}[res["density"]])
    a,b,c=st.columns(3); a.metric("Road Health",f"{score}/100"); b.metric("Highest Pothole Confidence",f"{res['pothole_conf']:.2f}"); c.metric("Severity",severity(res["pothole_conf"]) if res["potholes"] else "NONE")
    st.caption("Prototype prioritization score; not an engineering measurement.")

    st.markdown("### 🚶 Pedestrian Safety")
    a,b,c=st.columns(3); a.metric("Maximum Pedestrians",res["max_people"]); b.metric("Nearby Vehicles",res["max_nearby"]); c.metric("Sustained Proximity Risk","HIGH" if res["ped_risk"] else "LOW")
    if res["ped_risk"]: st.warning("⚠️ Sustained person–vehicle proximity was detected. Review the original footage.")

    st.markdown("### 🏗️ Infrastructure / Road Scene Intelligence")
    a,b=st.columns(2); a.metric("Traffic Signs Detected",res["signs"]); b.metric("Road-Marking Observations",res["marking_hits"])
    st.info("A camera can detect visible signs/markings, but a missing divider or missing/faded zebra crossing cannot be reliably proven from absence in one frame. Such cases require scene context or a dedicated trained deficiency model.")

    st.markdown("### 💧 Waterlogging / Road Hazard")
    if res["water_hits"]:
        st.warning(f"⚠️ {res['water_hits']} water-like road-surface observations were generated by a visual heuristic. These are review candidates, not measured water depth.")
    else: st.success("No water-like road-surface candidate was generated.")

    st.markdown("### 🚨 Incident Intelligence")
    if res["motion_events"]: st.warning(f"⚠️ {res['motion_events']} abnormal vehicle-motion candidate(s) detected from tracked trajectories.")
    else: st.success("No temporally confirmed abnormal-motion candidate was generated.")
    st.caption("This is not a legal rash-driving or hit-and-run determination.")

    st.markdown("### 🔎 Vehicle Identification / ANPR")
    if res["plates"]:
        st.write(f"License plates detected: **{res['plates']}**")
        if res["ocr"]:
            for x in res["ocr"][:10]: st.write(f"Candidate **{x['text']}** | OCR confidence **{x['confidence']:.2f}** | frame **{x['frame']}**")
        else: st.warning("Plates were detected, but no sufficiently confident OCR result was produced. No registration number was invented.")
    else: st.info("No license plate was detected in the processed frames.")

    st.markdown("### 📡 AI Event Feed")
    if res["events"]:
        st.dataframe([{"Time":e["timestamp"],"Video time (s)":e["video_time_s"],"Event":e["event_type"],"Confidence":e["confidence"],"Details":str(e["details"])} for e in res["events"][-40:]],use_container_width=True,hide_index=True)
    else: st.success("No event was generated from the processed camera footage.")

    st.markdown("### 🤖 AI Detection Output")
    if os.path.exists(res["output"]): st.video(res["output"])

    st.markdown("### 🗺️ City Intelligence Map")
    gps=[e for e in res["events"] if "latitude" in e and e.get("latitude") is not None]
    if not gps:
        st.info("No GPS was supplied with this camera input, so UrbanSense will not fabricate map locations.")
    else:
        m=folium.Map(location=[gps[0]["latitude"],gps[0]["longitude"]],zoom_start=14)
        pts=[]
        for e in gps:
            folium.Marker([e["latitude"],e["longitude"]],tooltip=e["event_type"],popup=f"{e['event_type']} | confidence {e['confidence']}").add_to(m); pts.append([e["latitude"],e["longitude"],max(.2,e["confidence"])])
        HeatMap(pts,radius=30,blur=20,min_opacity=.4).add_to(m)
        st_folium(m,width=None,height=430)

    st.markdown("### 🚨 Recommended City Action")
    actions=0
    if res["potholes"]: st.warning("🛠️ Prioritize road-maintenance inspection for detected road defects."); actions+=1
    if res["ped_risk"]: st.warning("🚶 Review pedestrian safety conditions at the observed scene."); actions+=1
    if res["density"]=="HIGH": st.warning("🚦 Review traffic-management conditions for elevated vehicle density."); actions+=1
    if res["motion_events"]: st.warning("🚨 Review abnormal-motion candidates against the original footage."); actions+=1
    if not actions: st.success("No high-priority action candidate was generated.")

    st.markdown("---")
    st.markdown("### 🧠 UrbanSense Intelligence Pipeline")
    st.write("🎥 Camera / Bus Video → 🤖 Edge AI → 🔎 Real Observations → 📡 Event Detection → 🗺️ GIS → 🚨 City Action")
    st.caption("UrbanSense prototype. Results shown above are generated from the selected video; GPS is displayed only when supplied as input.")
