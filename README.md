# UrbanSense — SIH26124

AI-powered mobile urban intelligence using public-transport camera feeds.

## Pipeline

Camera → Edge AI → Detection & Tracking → Event Engine → GPS/GIS → City Action → Fleet Intelligence

## Modes

- **Presentation Demo Video:** real inference from the built-in road video plus clearly labelled demo scenarios for SIH capabilities not present in the short clip.
- **Upload Video:** demo scenarios are disabled; dashboard reports only evidence derived from the uploaded video.
- **Live Camera / RTSP / HTTP:** samples a live stream for a configurable window and produces the same event/dashboard pipeline.

## Real AI components

- Ultralytics YOLO + ByteTrack for vehicle/person detection and tracking
- Fine-tuned pothole detector
- Traffic-sign detector
- License-plate detector
- EasyOCR when plate crops are readable
- Video-derived density and hotspot analytics

## Candidate / prototype components

Waterlogging, traffic-signal state, pedestrian proximity and abnormal motion use transparent heuristics/candidates. Missing-object deficiencies and hit-and-run require dedicated scene/temporal models for production.

## GIS

- No GPS
- Fixed GPS
- Route-assigned demo GIS with interpolated positions

The demo route is explicitly not actual camera GPS.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; streamlit run app.py
```

Missing specialist weights are downloaded automatically from their documented public model sources when possible.

## Important credibility rule

Demo examples are never mixed into uploaded-video/live-camera mode.

## Deployment

The app is designed for Streamlit Community Cloud. For a production deployment, move large weights to a model registry/object store, add authentication, encrypted transport, signed events and secure evidence retention.
