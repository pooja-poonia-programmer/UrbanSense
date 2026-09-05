from pathlib import Path
import traceback
import pandas as pd
import streamlit as st
import config
import models
import video_pipeline
import gis_engine
import analytics
from event_engine import demo_events, action_for

st.set_page_config(page_title="UrbanSense | City Intelligence",page_icon="🚌",layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:3rem}
.hero{padding:24px 28px;border-radius:18px;background:linear-gradient(135deg,#0f172a,#1e293b);color:white;margin-bottom:18px}
.hero h1{font-size:3rem;margin:0}.hero p{font-size:1.1rem;opacity:.85}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.72rem;font-weight:700;margin:2px}
.real{background:#dcfce7;color:#166534}.demo{background:#fef3c7;color:#92400e}.cand{background:#dbeafe;color:#1e40af}
.card{border:1px solid #e2e8f0;border-radius:14px;padding:14px;background:#fff}
.small{color:#64748b;font-size:.85rem}
.issue-card{border:1px solid #334155;border-radius:12px;padding:13px 16px;margin:9px 0;background:linear-gradient(135deg,#0f172a,#111827);color:#e5e7eb;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.issue-title{font-size:1rem;font-weight:750;margin-bottom:8px;color:#fff}.issue-line{font-size:.84rem;line-height:1.55;margin:2px 0}.issue-line b{color:#f8fafc}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🚌 UrbanSense</h1>
<p>AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet</p>
<div>🟢 SYSTEM ONLINE &nbsp; | &nbsp; Camera → Edge AI → Event Engine → GIS → City Action</div>
</div>
""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI models…")
def get_bundle():
    return models.load_bundle()

with st.expander("🔐 Privacy, safety & credibility"):
    st.write("The prototype does not perform face recognition. In real mode, outputs come from the selected camera/video. Candidate heuristics are not legal determinations. Demo scenarios are visibly labelled.")

bundle=get_bundle()

st.subheader("🩺 AI System Status")
status_cols=st.columns(5)
for col,key in zip(status_cols,["yolo","pothole","sign","plate","ocr"]):
    s=bundle.statuses.get(key)
    with col:
        st.metric(s.name if s else key,"ONLINE" if s and s.available else "OPTIONAL OFF")
        if s and not s.available: st.caption(s.message[:110])

st.divider()

st.subheader("🎥 Camera / CCTV Input")
source_mode=st.radio(
    "Input source",
    ["Presentation Demo Video","Upload Video","Live Camera / RTSP / HTTP"],
    horizontal=True
)

source=None
is_demo=source_mode=="Presentation Demo Video"

if source_mode=="Presentation Demo Video":
    if config.DEMO_VIDEO.exists():
        source=str(config.DEMO_VIDEO)
        st.video(str(config.DEMO_VIDEO))
        st.info("🟡 PRESENTATION DEMO MODE — real AI results from the built-in video are shown, and unsupported SIH scenarios are supplemented with clearly labelled DEMO EXAMPLES.")
    else:
        st.error(f"Demo video missing: {config.DEMO_VIDEO}")
elif source_mode=="Upload Video":
    up=st.file_uploader("Upload MP4 / AVI / MOV / MKV",type=["mp4","avi","mov","mkv"])
    if up:
        path=config.OUTPUTS_DIR/"uploaded_input.mp4"
        path.write_bytes(up.getbuffer())
        source=str(path)
        st.video(up)
        st.success("🟢 REAL VIDEO MODE — demo events are disabled.")
else:
    st.info("For a physical IP camera, paste its RTSP/HTTP stream URL. For a camera attached to this computer, use camera index 0. The stream is sampled for a fixed window so the dashboard can finish and display results.")
    live_src=st.text_input("RTSP / HTTP URL or local camera index","0")
    live_seconds=st.slider("Live analysis window (seconds)",10,60,20)
    source=int(live_src) if live_src.strip().isdigit() else live_src.strip()

bus_id=st.text_input("Bus / Camera ID",config.DEMO_BUS_ID if is_demo else "BUS-LIVE-01")

st.subheader("📍 GIS / GPS Configuration")
gps_choice=st.radio("Location mode",["No GPS","Fixed GPS","Route-assigned GIS (demo mode)"],horizontal=True)

gis_mode="none"; fixed_point=None; route_start=None; route_end=None
if gps_choice=="Fixed GPS":
    gis_mode="fixed"
    c1,c2=st.columns(2)
    lat=c1.number_input("Latitude",value=float(config.DEFAULT_FIXED_GPS["lat"]),format="%.6f")
    lon=c2.number_input("Longitude",value=float(config.DEFAULT_FIXED_GPS["lon"]),format="%.6f")
    fixed_point=gis_engine.fixed(lat,lon)
    if not fixed_point: st.error("Invalid coordinates.")
elif gps_choice.startswith("Route"):
    gis_mode="route"
    c1,c2,c3,c4=st.columns(4)
    sl=c1.number_input("Start latitude",value=float(config.DEFAULT_ROUTE_START["lat"]),format="%.6f")
    so=c2.number_input("Start longitude",value=float(config.DEFAULT_ROUTE_START["lon"]),format="%.6f")
    el=c3.number_input("End latitude",value=float(config.DEFAULT_ROUTE_END["lat"]),format="%.6f")
    eo=c4.number_input("End longitude",value=float(config.DEFAULT_ROUTE_END["lon"]),format="%.6f")
    route_start,route_end=gis_engine.route(sl,so,el,eo)
    st.info("🟡 DEMO GIS — event positions are interpolated along the supplied route; this is not actual camera GPS.")

run=st.button("▶ RUN FULL AI ANALYSIS",type="primary",use_container_width=True)

if run:
    if source is None:
        st.error("Select or upload a video/camera source first.")
    elif bundle.yolo is None:
        st.error("Primary YOLO model is unavailable. The app cannot perform video AI.")
    else:
        bar=st.progress(0.0)
        def cb(x): bar.progress(min(1.0,max(0.0,x)))
        with st.spinner("Running edge AI, tracking, specialist models and event generation…"):
            result=video_pipeline.process(
                source,bundle,bus_id,config.OUTPUTS_DIR,
                live_seconds=(live_seconds if source_mode=="Live Camera / RTSP / HTTP" else None),
                progress=cb
            )
        bar.empty()
        st.session_state["result"]=result
        st.session_state["is_demo_run"]=is_demo
        st.session_state["gps_mode"]=gis_mode
        st.session_state["fixed_point"]=fixed_point
        st.session_state["route_start"]=route_start
        st.session_state["route_end"]=route_end

result=st.session_state.get("result")
is_demo_run=st.session_state.get("is_demo_run",False)
has_run=result is not None and result.ok
if has_run:
    real_events=list(result.events)
else:
    real_events=[]

# Add location to real events without ever throwing.
for e in real_events:
    try:
        p=gis_engine.point_for_time(
            st.session_state.get("gps_mode","none"),
            e.video_time or 0,
            result.duration if result else 0,
            st.session_state.get("fixed_point"),
            st.session_state.get("route_start"),
            st.session_state.get("route_end"),
        )
        if p:
            e.latitude,e.longitude=p.lat,p.lon
    except Exception:
        pass

display_events=real_events
if is_demo_run and has_run:
    # Demo fills capabilities not actually seen in the short clip.
    real_types={e.event_type for e in real_events}
    display_events=real_events+[e for e in demo_events() if e.event_type not in real_types]
elif is_demo_run and not has_run:
    display_events=demo_events()

# ---------------- Dashboard ----------------
st.divider()
st.subheader("🏙️ City Command Center")
if has_run:
    road_events=[e for e in real_events if e.event_type=="POTHOLE_DETECTED"]
    c=st.columns(5)
    c[0].metric("Peak Vehicles",result.peak)
    c[1].metric("Avg / Frame",result.average)
    c[2].metric("Road Defects",len(road_events))
    c[3].metric("AI Events",len(display_events))
    c[4].metric("Tracked IDs",result.unique_vehicle_ids)
else:
    d=demo_events()
    c=st.columns(5)
    c[0].metric("Peak Vehicles","5" if is_demo_run else "—")
    c[1].metric("Avg / Frame","3.0" if is_demo_run else "—")
    c[2].metric("Road Defects","2" if is_demo_run else "—")
    c[3].metric("AI Events",len(d) if is_demo_run else "—")
    c[4].metric("Tracked IDs","DEMO" if is_demo_run else "—")
    st.caption("Run the analysis to replace presentation placeholders with video-derived statistics.")

st.subheader("🚦 Traffic Intelligence")
if has_run:
    cc=result.class_counts
    c=st.columns(5)
    c[0].metric("Cars",cc.get("car",0)); c[1].metric("Buses",cc.get("bus",0)); c[2].metric("Trucks",cc.get("truck",0)); c[3].metric("Motorcycles",cc.get("motorcycle",0)); c[4].metric("Bicycles",cc.get("bicycle",0))
else:
    c=st.columns(5)
    for col,val in zip(c,["12","3","2","2","1"] if is_demo_run else ["—"]*5): col.metric("Demo" if is_demo_run else "Vehicles",val)
st.caption("Counts are frame-level detections; ByteTrack IDs provide temporary track identities.")

def rich_issue(title, location, severity, status, icon="📍", extra=""):
    return f"""
    <div class="issue-card">
      <div class="issue-title">{icon} {title}</div>
      <div class="issue-line">📍 <b>{location}</b></div>
      <div class="issue-line">⚠️ <b>Severity:</b> {severity}</div>
      {f'<div class="issue-line">{extra}</div>' if extra else ''}
      <div class="issue-line">🛠️ <b>Status:</b> {status}</div>
    </div>
    """

st.subheader("🛣️ Infrastructure Deficiency Intelligence")
st.caption("Prototype infrastructure observations for demonstrating city-level maintenance prioritization.")
real_infra = [e for e in display_events if e.event_type in {"POTHOLE_DETECTED", "ROAD_DEFECT_CANDIDATE", "TRAFFIC_SIGN_CANDIDATE"}]
if is_demo_run:
    c=st.columns(3)
    c[0].metric("Infrastructure Issues",3)
    c[1].metric("High Priority",2)
    c[2].metric("Maintenance Required",1)
    st.markdown("**Detected Infrastructure Issues**")
    st.markdown(rich_issue("Missing Road Divider","Sector 60","HIGH","Requires Inspection","📍"),unsafe_allow_html=True)
    st.markdown(rich_issue("Faded Zebra Crossing","Sector 61","MEDIUM","Maintenance Required","📍"),unsafe_allow_html=True)
    st.markdown(rich_issue("Damaged Traffic Signboard","Sector 62","HIGH","Requires Inspection","📍"),unsafe_allow_html=True)
    st.info("🟡 DEMO EXAMPLES — absence of a divider/zebra/signboard cannot be proven from a generic object detector alone. A dedicated deficiency model or scene-context verification is required in production.")
else:
    c=st.columns(3)
    c[0].metric("Infrastructure Events",len(real_infra))
    c[1].metric("High Priority",sum(1 for e in real_infra if e.severity=="HIGH"))
    c[2].metric("Maintenance Review",len(real_infra))
    if real_infra:
        st.markdown("**Detected Infrastructure Issues**")
        for e in real_infra:
            loc = f"{e.latitude:.5f}, {e.longitude:.5f}" if e.latitude is not None and e.longitude is not None else "Video location unavailable"
            title = {"POTHOLE_DETECTED":"Pothole / Damaged Road","ROAD_DEFECT_CANDIDATE":"Road Defect Candidate","TRAFFIC_SIGN_CANDIDATE":"Traffic Sign Candidate"}.get(e.event_type,e.event_type.replace('_',' ').title())
            st.markdown(rich_issue(title,loc,e.severity,"Review required", "📍", f"🎞️ <b>Video time:</b> {e.video_time:.1f}s"),unsafe_allow_html=True)
    else:
        st.info("No infrastructure event detected in the selected video/source.")

st.subheader("💧 Waterlogging & Road Hazard Intelligence")
st.caption("Visual waterlogging candidates and response prioritization. Water-depth values are not inferred unless a calibrated depth model is available.")
water=[e for e in display_events if e.event_type=="WATERLOGGING_CANDIDATE"]
if is_demo_run:
    c=st.columns(3)
    c[0].metric("Waterlogging Events",2)
    c[1].metric("High Severity",1)
    c[2].metric("Immediate Attention",1)
    st.markdown("**Active Road Hazards**")
    st.markdown(rich_issue("Severe Waterlogging","Sector 63","HIGH","Immediate Attention","💧","📏 <b>Estimated water depth:</b> 25–40 cm <span class='small'>(demo scenario)</span>"),unsafe_allow_html=True)
    st.markdown(rich_issue("Moderate Water Accumulation","Sector 59","MEDIUM","Monitor","💧","📏 <b>Estimated water depth:</b> 10–20 cm <span class='small'>(demo scenario)</span>"),unsafe_allow_html=True)
    st.info("🟡 DEMO EXAMPLES — water-depth estimates are illustrative. Production deployment needs calibrated visual depth estimation or external sensor data.")
elif water:
    c=st.columns(3)
    c[0].metric("Waterlogging Candidates",len(water))
    c[1].metric("High Severity",sum(1 for e in water if e.severity=="HIGH"))
    c[2].metric("Immediate Review",sum(1 for e in water if e.severity=="HIGH"))
    st.markdown("**Active Road Hazards**")
    for e in water:
        loc=f"{e.latitude:.5f}, {e.longitude:.5f}" if e.latitude is not None and e.longitude is not None else "Video location unavailable"
        st.markdown(rich_issue("Waterlogging Candidate",loc,e.severity,"Manual verification required","💧",f"🎞️ <b>Video time:</b> {e.video_time:.1f}s"),unsafe_allow_html=True)
else:
    st.info("No waterlogging candidate detected in the selected video/source.")

st.subheader("🚨 Incident Intelligence")
st.caption("Incident workflow for candidate detection, vehicle tracking, alert generation and investigation support.")
rash=[e for e in display_events if e.event_type=="ABNORMAL_MOTION_CANDIDATE"]
hit=[e for e in display_events if e.event_type=="HIT_AND_RUN_CANDIDATE"]
if is_demo_run:
    c=st.columns(3)
    c[0].metric("Active Incidents",2)
    c[1].metric("Critical Incidents",1)
    c[2].metric("Alerts Generated",1)
    st.markdown("**Incident Alerts**")
    st.markdown(rich_issue("Suspected Rash Driving — INC-001","Sector 60","HIGH","Alert Generated","🚨","🚗 <b>Tracked Vehicle:</b> TRACK-27 &nbsp; 🚌 <b>Reporting Bus:</b> BUS-101<br>🕐 <b>Timestamp:</b> 2026-09-04 18:42:15<br>📊 <b>Prototype confidence:</b> 87%"),unsafe_allow_html=True)
    st.markdown(rich_issue("Suspected Hit-and-Run — INC-002","Sector 61","CRITICAL","Requires Investigation","🚨","🚗 <b>Tracked Vehicle:</b> TRACK-42 &nbsp; 🚌 <b>Reporting Bus:</b> BUS-102<br>🕐 <b>Timestamp:</b> 2026-09-04 18:47:32<br>📊 <b>Prototype confidence:</b> 81%"),unsafe_allow_html=True)
    st.info("🟡 DEMO EXAMPLES — rash-driving and hit-and-run classification are simulated. The real workflow uses tracked IDs, timestamps, GPS metadata and evidence for manual review.")
else:
    incidents=rash+hit
    c=st.columns(3)
    c[0].metric("Incident Candidates",len(incidents))
    c[1].metric("High/Critical",sum(1 for e in incidents if e.severity in {"HIGH","CRITICAL"}))
    c[2].metric("Manual Review",len(incidents))
    if incidents:
        st.markdown("**Incident Alerts**")
        for e in incidents:
            title="Rash-driving Candidate" if e.event_type=="ABNORMAL_MOTION_CANDIDATE" else "Hit-and-run Candidate"
            loc=f"{e.latitude:.5f}, {e.longitude:.5f}" if e.latitude is not None and e.longitude is not None else "GPS unavailable"
            st.markdown(rich_issue(title,loc,e.severity,"Manual Investigation","🚨",f"🚗 <b>Tracked Vehicle:</b> {e.track_id or 'Not associated'}<br>🕐 <b>Video time:</b> {e.video_time:.1f}s<br>📊 <b>Candidate confidence:</b> {e.confidence:.0%}"),unsafe_allow_html=True)
    else:
        st.success("No incident candidate detected in the selected video/source.")
st.caption("Incident candidates are not legal determinations. Hit-and-run requires collision + departure evidence over time.")

st.subheader("🔎 Vehicle Identification & ANPR Intelligence")
st.caption("Number-plate localization and OCR workflow. A plate must be clearly visible for reliable recognition.")
plates=[e for e in display_events if e.event_type=="PLATE_DETECTED"]
if is_demo_run:
    c=st.columns(3)
    c[0].metric("Vehicles Identified",2)
    c[1].metric("High OCR Confidence",1)
    c[2].metric("Review Required",1)
    st.markdown("**Vehicle Identification Records**")
    st.markdown(rich_issue("Vehicle: TRACK-27","Sector 60","HIGH","Verified Candidate","🚗","🔢 <b>Registration Number:</b> CH01AB1234<br>📊 <b>OCR Confidence:</b> 94%<br>🚨 <b>Linked Incident:</b> INC-001<br>🕐 <b>Timestamp:</b> 2026-09-04 18:42:15"),unsafe_allow_html=True)
    st.markdown(rich_issue("Vehicle: TRACK-42","Sector 61","MEDIUM","Review Required","🚗","🔢 <b>Registration Number:</b> PB65XY7890<br>📊 <b>OCR Confidence:</b> 89%<br>🚨 <b>Linked Incident:</b> INC-002<br>🕐 <b>Timestamp:</b> 2026-09-04 18:47:32"),unsafe_allow_html=True)
    st.info("🟡 DEMO EXAMPLES — registration numbers above are simulated. Real mode displays only plate detections/OCR produced from the selected video.")
elif plates:
    c=st.columns(3)
    c[0].metric("Plate Detections",len(plates))
    c[1].metric("Readable Plates",sum(1 for e in plates if e.details.get("plate_text")))
    c[2].metric("Review Required",sum(1 for e in plates if not e.details.get("plate_text")))
    st.markdown("**Vehicle Identification Records**")
    for e in plates:
        plate=e.details.get("plate_text") or "Plate detected — OCR low confidence"
        conf=e.details.get("ocr_confidence")
        st.markdown(rich_issue(f"Vehicle: {e.track_id or 'Unassociated'}",f"{e.latitude:.5f}, {e.longitude:.5f}" if e.latitude is not None and e.longitude is not None else "Video location unavailable",e.severity,"Verified candidate" if conf else "Review Required","🚗",f"🔢 <b>Registration:</b> {plate}<br>📊 <b>OCR Confidence:</b> {conf:.0%}" if conf else "🔎 <b>OCR:</b> no reliable text"),unsafe_allow_html=True)
else:
    c=st.columns(3)
    c[0].metric("Vehicles Identified",0)
    c[1].metric("High OCR Confidence",0)
    c[2].metric("Review Required",0)
    st.info("No number plate detected in the selected video/source.")

st.subheader("📋 Live AI Event Feed")
if display_events:
    st.dataframe(pd.DataFrame([{
        "Event ID":e.event_id,"Time":e.video_time,"Event":e.event_type,
        "Confidence":e.confidence,"Severity":e.severity,"Bus":e.bus_id,
        "Track":e.track_id,"Latitude":e.latitude,"Longitude":e.longitude,"Source":e.source,"Status":e.label
    } for e in display_events]),use_container_width=True,hide_index=True)
else:
    st.info("No events generated.")

st.subheader("🎬 AI Detection Output")
if has_run:
    if result.playable_output and Path(result.playable_output).exists():
        st.video(str(result.playable_output))
    else:
        st.error(result.video_error or "Annotated video unavailable.")
else:
    st.info("Run analysis to generate the annotated video.")

st.subheader("🗺️ City Intelligence Map + 🔥 Congestion Heatmap")
if st.session_state.get("gps_mode")=="none":
    if is_demo_run:
        st.info("Select Route-assigned GIS or Fixed GPS to display the geographic visualization.")
    else:
        st.info("No GPS supplied. The dashboard remains fully functional; add GPS to spatially place events.")
else:
    try:
        heat=gis_engine.heat_points(
            result.density_samples if has_run else [],
            st.session_state["gps_mode"],
            st.session_state.get("fixed_point"),
            st.session_state.get("route_start"),
            st.session_state.get("route_end"),
        )
        fmap,msg=gis_engine.render(
            st.session_state["gps_mode"],
            st.session_state.get("fixed_point"),
            st.session_state.get("route_start"),
            st.session_state.get("route_end"),
            heat,display_events
        )
        if fmap is not None:
            from streamlit_folium import st_folium
            st_folium(fmap,width=None,height=520,returned_objects=[])
        else:
            st.warning(msg)
            st.dataframe(pd.DataFrame([{
                "Event":e.event_type,"Latitude":e.latitude,"Longitude":e.longitude,"Status":e.label
            } for e in display_events]),use_container_width=True,hide_index=True)
    except Exception as exc:
        st.warning(f"GIS visualization unavailable: {exc}")
        st.dataframe(pd.DataFrame([{
            "Event":e.event_type,"Latitude":e.latitude,"Longitude":e.longitude,"Status":e.label
        } for e in display_events]),use_container_width=True,hide_index=True)

st.subheader("🚌 Fleet Intelligence")
fleet=analytics.fleet_summary(display_events)
if fleet:
    st.dataframe(pd.DataFrame(fleet),use_container_width=True,hide_index=True)
else:
    st.info("No fleet events yet.")

st.subheader("📈 Route / Bottleneck / OD Analytics")
if has_run:
    hot=analytics.hotspot_segments(result.density_samples)
    if hot: st.dataframe(pd.DataFrame(hot),use_container_width=True,hide_index=True)
    delay=max(0,round((result.peak-2)*.7,1))
    st.metric("Prototype route-delay estimate",f"{delay} min")
else:
    st.metric("Prototype route-delay estimate","4.2 min" if is_demo_run else "—")
st.info("OD analytics requires multi-trip GPS history. This prototype intentionally does not fabricate a historical OD matrix.")

st.subheader("📡 Edge AI & Bandwidth Minimization")
if has_run:
    event_rate=round(len(real_events)/max(1,result.frames)*100,2)
    c=st.columns(3); c[0].metric("Frames processed",result.frames); c[1].metric("Real events",len(real_events)); c[2].metric("Event rate",f"{event_rate}%")
else:
    c=st.columns(3); c[0].metric("Edge inference","LOCAL"); c[1].metric("Transmission","EVENT ONLY"); c[2].metric("Raw video","NOT REQUIRED")
st.write("Camera → local inference → confidence/event filter → compact event metadata + evidence → city backend. This is the bandwidth-minimization architecture.")

st.subheader("🏛️ Recommended City Action")
if display_events:
    unique={}
    for e in display_events:
        unique.setdefault(e.event_type,e)
    for e in list(unique.values())[:10]:
        st.write(f"**{e.event_type.replace('_',' ').title()}** → {action_for(e.event_type)}  {'🟡 DEMO EXAMPLE' if e.is_demo else '🟢 VIDEO DERIVED'}")
else:
    st.info("No action candidate.")


st.markdown("---")
st.caption("UrbanSense SIH26124 • Prototype • REAL AI and VIDEO DERIVED outputs are separated from DEMO EXAMPLES and CANDIDATE heuristics.")
