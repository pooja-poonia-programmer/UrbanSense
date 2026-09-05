from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from utils import now_id
import config

@dataclass
class Event:
    event_type: str
    confidence: float
    severity: str
    bus_id: str
    source: str
    video_time: Optional[float] = None
    track_id: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    details: dict = field(default_factory=dict)
    evidence_path: Optional[str] = None
    timestamp: str = ""
    is_demo: bool = False
    label: str = config.REAL_TAG
    event_id: str = field(default_factory=lambda: now_id("EVT"))

    def __post_init__(self):
        if not self.timestamp:
            import datetime
            self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.confidence = round(float(self.confidence), 2)

    def as_dict(self):
        return asdict(self)

def severity(conf):
    return "HIGH" if conf >= .75 else "MEDIUM" if conf >= .5 else "LOW"

class EventEngine:
    def __init__(self):
        self.events = []
        self.last = {}

    def add(self, event_type, confidence, bus_id, source, video_time=None,
            track_id=None, details=None, evidence_path=None, is_demo=False,
            label=None, cooldown=True):
        t = float(video_time or 0)
        key = (event_type, track_id)
        if cooldown and key in self.last and t - self.last[key] < config.EVENT_COOLDOWN_SECONDS:
            return None
        self.last[key] = t
        e = Event(
            event_type=event_type,
            confidence=confidence,
            severity=severity(confidence),
            bus_id=bus_id,
            source=source,
            video_time=t,
            track_id=track_id,
            details=details or {},
            evidence_path=evidence_path,
            is_demo=is_demo,
            label=label or (config.DEMO_TAG if is_demo else config.CANDIDATE_TAG if "CANDIDATE" in event_type else config.REAL_TAG),
        )
        self.events.append(e)
        return e

    def all(self):
        return list(self.events)

def demo_events():
    specs = [
        ("POTHOLE_DETECTED", .87, "pothole model", {"severity":"HIGH"}),
        ("WATERLOGGING_CANDIDATE", .82, "presentation scenario", {"scenario":"urban waterlogging"}),
        ("TRAFFIC_SIGN_CANDIDATE", .84, "presentation scenario", {"issue":"damaged/faded signboard"}),
        ("TRAFFIC_SIGNAL_CANDIDATE", .80, "presentation scenario", {"issue":"signal-state intelligence"}),
        ("ROAD_DEFECT_CANDIDATE", .78, "presentation scenario", {"issue":"missing zebra crossing"}),
        ("ROAD_DEFECT_CANDIDATE", .80, "presentation scenario", {"issue":"missing road divider"}),
        ("PEDESTRIAN_PROXIMITY", .79, "presentation scenario", {"scenario":"vulnerable road user"}),
        ("ABNORMAL_MOTION_CANDIDATE", .81, "presentation scenario", {"scenario":"rash driving"}),
        ("HIT_AND_RUN_CANDIDATE", .79, "presentation workflow", {"scenario":"collision + departure review"}),
        ("PLATE_DETECTED", .91, "presentation scenario", {"plate_text":"DEMO-PLATE-01","ocr_confidence":.91}),
        ("CONGESTION_HOTSPOT", .83, "presentation scenario", {"vehicles":8}),
    ]
    out=[]
    for i,(typ,conf,src,details) in enumerate(specs):
        out.append(Event(
            event_type=typ, confidence=conf, severity=severity(conf),
            bus_id=config.DEMO_BUS_ID, source=src, video_time=2+i*2.5,
            details=details, is_demo=True, label=config.DEMO_TAG
        ))
    return out

def action_for(event_type):
    return config.ACTIONS.get(event_type, "Log for city-intelligence review.")
