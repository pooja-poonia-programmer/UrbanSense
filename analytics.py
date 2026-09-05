import config

def hotspot_segments(samples, bin_seconds=5):
    bins={}
    for s in samples:
        b=int(float(s["video_time"])//bin_seconds)
        bins.setdefault(b,[]).append(int(s["vehicle_count"]))
    rows=[]
    for b,vals in bins.items():
        rows.append({
            "Start (s)":round(b*bin_seconds,1),
            "End (s)":round((b+1)*bin_seconds,1),
            "Average vehicles":round(sum(vals)/len(vals),2),
            "Peak vehicles":max(vals),
        })
    return sorted(rows,key=lambda x:x["Average vehicles"],reverse=True)

def fleet_summary(events):
    d={}
    for e in events:
        d.setdefault(e.bus_id,[]).append(e)
    return [
        {"Bus ID":bid,"Events":len(es),"Event types":", ".join(sorted(set(e.event_type for e in es)))}
        for bid,es in d.items()
    ]

def coverage():
    return [
        ("Potholes / damaged roads","Pothole detector","REAL AI"),
        ("Vehicle detection / classification / counting","YOLO + frame analytics","REAL AI"),
        ("Vehicle tracking","ByteTrack IDs","REAL AI"),
        ("Traffic density / bottlenecks","Per-frame density + hotspot bins","VIDEO DERIVED"),
        ("Pedestrian safety","Person detection + proximity candidate","CANDIDATE"),
        ("Traffic signs","Traffic-sign model","CANDIDATE"),
        ("Traffic signals","Visual signal-color candidate","CANDIDATE"),
        ("Waterlogging","Visual heuristic candidate","CANDIDATE"),
        ("Missing divider","Scene-context workflow","DEMO ONLY / MODEL REQUIRED"),
        ("Missing zebra crossing","Scene-context workflow","DEMO ONLY / MODEL REQUIRED"),
        ("Damaged/missing signboard","Scene-context workflow","DEMO ONLY / MODEL REQUIRED"),
        ("Rash driving","Temporal abnormal-motion candidate","CANDIDATE"),
        ("Hit-and-run","Collision/departure workflow","DEMO ONLY / MODEL REQUIRED"),
        ("ANPR","Plate detector + OCR when readable","REAL DETECTOR / OCR CANDIDATE"),
        ("GPS / GIS","Fixed GPS or route interpolation","DEMO/USER SUPPLIED"),
        ("Congestion heatmap","Video density + GIS route","VIDEO DERIVED"),
        ("Fleet aggregation","Event schema grouped by bus","VIDEO/SESSION DERIVED"),
        ("Route delay","Density-derived prototype estimate","VIDEO DERIVED"),
        ("OD patterns","Requires multi-trip GPS history","REQUIRES GPS HISTORY"),
        ("Edge AI / bandwidth","Local inference + event filtering","ARCHITECTURE"),
        ("City action","Event-to-action rules","RULE BASED"),
    ]
