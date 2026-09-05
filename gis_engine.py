from __future__ import annotations
from typing import Optional
from utils import LatLon, interpolate

def fixed(lat, lon):
    p = LatLon(float(lat), float(lon))
    return p if p.valid() else None

def route(sl, so, el, eo):
    a, b = LatLon(float(sl), float(so)), LatLon(float(el), float(eo))
    return (a,b) if a.valid() and b.valid() else (None,None)

def point_for_time(mode, video_time, duration, fixed_point=None, route_start=None, route_end=None):
    if mode == "fixed" and fixed_point and fixed_point.valid():
        return fixed_point
    if mode == "route" and route_start and route_end:
        t = (video_time or 0) / max(.1, duration or 1)
        return interpolate(route_start, route_end, t)
    return None

def heat_points(density_samples, mode, fixed_point=None, route_start=None, route_end=None):
    out=[]
    for s in density_samples:
        p=point_for_time(mode,s["video_time"],s.get("duration",0),fixed_point,route_start,route_end)
        if p:
            out.append([p.lat,p.lon,max(1,s["vehicle_count"])])
    return out

def render(mode, fixed_point=None, route_start=None, route_end=None, heat=None, events=None):
    try:
        import folium
        from folium.plugins import HeatMap
        if mode == "fixed" and fixed_point:
            center=[fixed_point.lat,fixed_point.lon]
        elif mode == "route" and route_start and route_end:
            center=[(route_start.lat+route_end.lat)/2,(route_start.lon+route_end.lon)/2]
        else:
            return None, "No GPS/location supplied."
        fmap=folium.Map(location=center,zoom_start=13,tiles="OpenStreetMap")
        if mode=="route":
            folium.PolyLine(
                [[route_start.lat,route_start.lon],[route_end.lat,route_end.lon]],
                tooltip="DEMO ROUTE — INTERPOLATED GPS",weight=5
            ).add_to(fmap)
            folium.Marker([route_start.lat,route_start.lon],tooltip="Route start").add_to(fmap)
            folium.Marker([route_end.lat,route_end.lon],tooltip="Route end").add_to(fmap)
        elif mode=="fixed":
            folium.Marker(center,tooltip="Fixed GPS").add_to(fmap)
        if heat:
            HeatMap(heat,radius=22,blur=18,min_opacity=.3).add_to(fmap)
        for e in events or []:
            if e.latitude is None or e.longitude is None: continue
            popup=f"{e.event_type} | {e.confidence:.2f} | {e.label}"
            folium.CircleMarker(
                [e.latitude,e.longitude],radius=6,popup=popup,fill=True,fill_opacity=.85
            ).add_to(fmap)
        return fmap, "OK"
    except Exception as exc:
        return None, f"GIS visualization unavailable: {exc}"
