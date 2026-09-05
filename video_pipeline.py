from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time, math
import cv2
import numpy as np
import config
from utils import resize_keep_aspect, center, distance, convert_video
from event_engine import EventEngine

@dataclass
class VideoResult:
    ok: bool
    message: str
    raw_output: Path|None=None
    playable_output: Path|None=None
    video_error: str|None=None
    frames: int=0
    fps: float=0
    duration: float=0
    peak: int=0
    average: float=0
    class_counts: dict=field(default_factory=dict)
    unique_vehicle_ids: int=0
    pedestrian_frames: int=0
    density_samples: list=field(default_factory=list)
    events: list=field(default_factory=list)
    notices: list=field(default_factory=list)

def _water_score(frame):
    h,w=frame.shape[:2]
    roi=frame[int(h*.58):h]
    if roi.size==0:return 0
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    sat=hsv[:,:,1].astype(np.float32); val=hsv[:,:,2].astype(np.float32)
    mask=(sat<65)&(val>105)
    frac=float(mask.mean())
    return max(0,min(1,(frac-.18)/.42))

def _signal_score(frame):
    h,w=frame.shape[:2]
    roi=frame[:int(h*.45)]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # Red/yellow/green luminous blob candidate; never called confirmed signal.
    masks=[]
    for lo,hi in [((0,100,100),(12,255,255)),((18,90,100),(38,255,255)),((40,70,90),(90,255,255))]:
        masks.append(cv2.inRange(hsv,np.array(lo),np.array(hi)))
    area=sum(int(np.count_nonzero(m)) for m in masks)
    return min(1.0,area/max(1,roi.shape[0]*roi.shape[1])/.025)

def process(source, bundle, bus_id, output_dir, live_seconds=None, progress=None):
    out=VideoResult(False,"")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        return VideoResult(False,"Could not open video/camera/RTSP source.")

    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_hint=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    is_live=live_seconds is not None
    start=time.time()
    raw=output_dir/f"urbansense_{int(time.time())}_raw.mp4"
    writer=None
    engine=EventEngine()
    class_counts={v:0 for v in ["car","motorcycle","bus","truck","bicycle"]}
    density=[]
    track_history={}
    seen=set()
    ped_frames=0
    per_frame=[]
    frame_idx=0

    try:
        while True:
            if is_live and time.time()-start >= live_seconds: break
            ok,frame=cap.read()
            if not ok: break
            frame_idx+=1
            if not is_live and frame_idx>config.MAX_FILE_FRAMES:
                out.notices.append(f"Stopped at {config.MAX_FILE_FRAMES} frames for safety.")
                break
            frame=resize_keep_aspect(frame,config.TARGET_MAX_DIM)
            h,w=frame.shape[:2]
            if writer is None:
                writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

            t=frame_idx/fps
            vehicles=[]
            people=[]
            try:
                r=bundle.yolo.track(
                    frame,persist=True,tracker="bytetrack.yaml",
                    conf=config.YOLO_CONF,verbose=False
                )[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        cid=int(box.cls[0]); name=r.names.get(cid,str(cid)).lower()
                        conf=float(box.conf[0]); xy=list(map(int,box.xyxy[0].tolist()))
                        tid=int(box.id[0]) if box.id is not None else None
                        label=f"{name} {conf:.2f}"
                        if tid is not None: label+=f" ID:{tid}"
                        if name=="person":
                            people.append(xy)
                            color=(60,180,75)
                        elif name in config.VEHICLE_NAMES:
                            vehicles.append((xy,name,conf,tid))
                            class_counts[name]=class_counts.get(name,0)+1
                            if tid is not None: seen.add(tid)
                            color=(50,130,240)
                            if tid is not None:
                                c=center(xy); hist=track_history.setdefault(tid,[])
                                if hist:
                                    d=distance(c,hist[-1])/max(1,math.hypot(w,h))
                                    if d>config.MOTION_RATIO:
                                        engine.add("ABNORMAL_MOTION_CANDIDATE",min(.85,.45+d*4),
                                            bus_id,"temporal motion heuristic",t,tid,
                                            {"note":"Camera-motion-sensitive candidate; manual review required."})
                                hist.append(c)
                                if len(hist)>8: hist.pop(0)
                        else:
                            continue
                        cv2.rectangle(frame,(xy[0],xy[1]),(xy[2],xy[3]),color,2)
                        cv2.putText(frame,label,(xy[0],max(18,xy[1]-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            except Exception as e:
                out.notices.append(f"YOLO/tracking issue: {e}")

            vc=len(vehicles); per_frame.append(vc)
            if people: ped_frames+=1
            density.append({"video_time":round(t,2),"vehicle_count":vc})

            # Pedestrian proximity based on current frame boxes.
            try:
                for p in people:
                    pc=center(p)
                    for vb,_,_,tid in vehicles:
                        if distance(pc,center(vb))/max(1,math.hypot(w,h)) < config.PROXIMITY_RATIO:
                            engine.add("PEDESTRIAN_PROXIMITY",.72,bus_id,"current-frame proximity",t,tid,
                                       {"note":"Pedestrian close to detected vehicle; candidate safety event."})
                            break
            except Exception as e:
                out.notices.append(f"Pedestrian heuristic issue: {e}")

            if frame_idx % config.SPECIALIST_EVERY == 0:
                if bundle.pothole is not None:
                    try:
                        r=bundle.pothole.predict(frame,conf=config.POTHOLE_CONF,verbose=False)[0]
                        if r.boxes is not None:
                            for b in r.boxes:
                                cf=float(b.conf[0]); xy=b.xyxy[0].tolist()
                                engine.add("POTHOLE_DETECTED",cf,bus_id,"pothole_model",t,
                                           details={"bbox":[round(x,1) for x in xy]})
                                cv2.rectangle(frame,(int(xy[0]),int(xy[1])),(int(xy[2]),int(xy[3])),(0,0,255),3)
                                cv2.putText(frame,f"POTHOLE {cf:.2f}",(int(xy[0]),max(20,int(xy[1])-7)),
                                            cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                    except Exception as e: out.notices.append(f"Pothole issue: {e}")

                if bundle.sign is not None:
                    try:
                        r=bundle.sign.predict(frame,conf=config.SIGN_CONF,verbose=False)[0]
                        if r.boxes is not None:
                            for b in r.boxes:
                                cf=float(b.conf[0]); xy=b.xyxy[0].tolist()
                                cid=int(b.cls[0]); name=r.names.get(cid,str(cid))
                                engine.add("TRAFFIC_SIGN_CANDIDATE",cf,bus_id,"traffic_sign_model",t,
                                           details={"class":name})
                                cv2.rectangle(frame,(int(xy[0]),int(xy[1])),(int(xy[2]),int(xy[3])),(0,165,255),2)
                    except Exception as e: out.notices.append(f"Sign model issue: {e}")

                if bundle.plate is not None:
                    try:
                        r=bundle.plate.predict(frame,conf=config.PLATE_CONF,verbose=False)[0]
                        if r.boxes is not None:
                            for b in r.boxes:
                                cf=float(b.conf[0]); xy=list(map(int,b.xyxy[0].tolist()))
                                x1,y1=max(0,xy[0]),max(0,xy[1]); x2,y2=min(w,xy[2]),min(h,xy[3])
                                crop=frame[y1:y2,x1:x2]
                                text=""; oc=0.0
                                if bundle.ocr is not None and crop.size and crop.shape[0]*crop.shape[1]>=900:
                                    try:
                                        rr=bundle.ocr.readtext(crop,detail=1,paragraph=False)
                                        if rr:
                                            best=max(rr,key=lambda x:float(x[2])); text=str(best[1]); oc=float(best[2])
                                    except Exception as e: out.notices.append(f"OCR issue: {e}")
                                details={"plate_text":text if oc>=config.OCR_CONF else None,
                                         "ocr_confidence":round(oc,2),
                                         "note":"Plate detected; OCR low confidence." if oc<config.OCR_CONF else "OCR readable."}
                                engine.add("PLATE_DETECTED",cf,bus_id,"license_plate_model",t,details=details)
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(180,20,255),2)
                                cv2.putText(frame, text if oc>=config.OCR_CONF else "PLATE / OCR LOW",
                                            (x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,20,255),2)
                    except Exception as e: out.notices.append(f"Plate model issue: {e}")

                ws=_water_score(frame)
                if ws>=.42:
                    engine.add("WATERLOGGING_CANDIDATE",ws,bus_id,"visual water heuristic",t,
                               details={"note":"Heuristic candidate, not a trained waterlogging model."})
                ss=_signal_score(frame)
                if ss>=.55:
                    engine.add("TRAFFIC_SIGNAL_CANDIDATE",ss,bus_id,"signal-color visual heuristic",t,
                               details={"note":"Candidate signal-state visual cue; not a traffic-law determination."})

            cv2.rectangle(frame,(0,0),(w,42),(20,20,20),-1)
            cv2.putText(frame,f"UrbanSense | {t:.1f}s | vehicles={vc} | pedestrians={len(people)}",
                        (10,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2)
            writer.write(frame)
            if progress and frame_hint:
                progress(min(1,frame_idx/frame_hint))
    finally:
        cap.release()
        if writer is not None: writer.release()

    if frame_idx==0:
        return VideoResult(False,"No readable frames were received from the source.")

    if max(per_frame,default=0)>=4:
        engine.add("CONGESTION_HOTSPOT",.70,bus_id,"video density analytics",
                   frame_idx/fps,details={"peak_vehicles":max(per_frame)})
    out.ok=True
    out.message="AI analysis completed successfully."
    out.raw_output=raw
    out.frames=frame_idx; out.fps=fps; out.duration=frame_idx/fps
    out.peak=max(per_frame,default=0)
    out.average=round(sum(per_frame)/len(per_frame),2) if per_frame else 0
    out.class_counts=class_counts
    out.unique_vehicle_ids=len(seen)
    out.pedestrian_frames=ped_frames
    for d in density: d["duration"]=out.duration
    out.density_samples=density
    out.events=engine.all()
    playable=output_dir/f"{raw.stem}_playable.mp4"
    ok,msg=convert_video(raw,playable)
    if ok:
        out.playable_output=playable
    else:
        out.video_error=msg
    return out
