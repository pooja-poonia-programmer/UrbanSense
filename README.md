# 🚀 UrbanSense

### AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/AI-YOLO-111111?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![GitHub](https://img.shields.io/badge/Repository-GitHub-181717?logo=github&logoColor=white)](https://github.com/)
[![Status](https://img.shields.io/badge/Status-Prototype-success)]()

> **UrbanSense turns public buses into mobile AI sensing nodes that continuously transform road-camera footage into structured traffic, road-safety and urban-intelligence events.**

🌐 **Live Demo:** [UrbanSense on Streamlit](https://urbansense.streamlit.app/)

---

# ✨ Overview

UrbanSense is an AI-powered urban intelligence prototype designed around a simple idea:

> **Every public bus can become a moving sensor for the city.**

Public transport vehicles already travel through major roads and can carry cameras capable of continuously observing the surrounding environment. Instead of treating this footage only as passive video recordings, UrbanSense applies computer vision and edge-oriented event processing to convert observations into actionable urban intelligence.

The prototype processes road-camera footage and provides:

- 🚗 Vehicle detection and classification
- 🚌 Bus, car and truck counting
- 🎯 Temporary vehicle tracking using ByteTrack
- 🛣️ Pothole and road-defect detection
- 🚶 Pedestrian proximity/risk analysis
- 🚦 Traffic and congestion intelligence
- 🚧 Infrastructure-deficiency workflows
- 💧 Waterlogging and road-hazard workflows
- 🚨 Incident-intelligence workflows
- 🔎 Number-plate detection and OCR workflow
- 📋 Structured AI event generation
- 📍 GPS/GIS-based event positioning
- 🔥 Congestion heatmap visualization
- 🚌 Fleet-level event aggregation
- 📈 Route and bottleneck analytics
- 📡 Edge-AI / bandwidth-minimization architecture
- 🏛️ Recommended city actions

UrbanSense is currently a **prototype for SIH26124**. Real AI/video-derived results are separated from prototype/demo scenarios wherever the available models cannot reliably establish a real-world event.

---

# 🎯 Problem Statement

## Existing Problem

Urban roads are dynamic environments containing:

- Vehicles
- Pedestrians
- Road defects
- Traffic congestion
- Waterlogging
- Traffic signs
- Missing or damaged infrastructure
- Potential traffic incidents

Although public buses frequently travel through these roads and may have onboard cameras, the resulting footage is commonly treated primarily as recorded video rather than as a continuous source of machine-readable urban intelligence.

## Limitations of Conventional Approaches

Traditional monitoring can depend heavily on:

- Fixed CCTV infrastructure
- Manual video inspection
- Citizen complaints
- Periodic road surveys
- Isolated traffic monitoring systems

These approaches can make continuous, city-wide road observation difficult and can delay the conversion of visual observations into actionable information.

## Proposed Solution

UrbanSense introduces an **AI-powered mobile sensing architecture** where public buses act as moving observation platforms.

```text
Public Bus Camera
       ↓
Edge AI / Computer Vision
       ↓
Object Detection + Tracking
       ↓
Event Generation
       ↓
GPS / GIS Association
       ↓
Urban Intelligence Dashboard
       ↓
Recommended City Action
