# AI-Based Real-Time Video Analytics System


## Project Description

A desktop-based Computer Vision application for detecting, tracking, and analyzing people and vehicles in real-time and recorded video.

The system uses YOLO-based object detection and ByteTrack multi-object tracking to maintain object identities across video frames. It provides real-time analytics including object counting, line-crossing analysis, zone occupancy, zone entry/exit, dwell-time measurement, and trajectory visualization through an interactive PySide6 desktop interface.

The application is designed for local/offline video processing and supports both live webcam input and recorded video files.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video processing and computer vision |
| Ultralytics YOLO | Object detection |
| ByteTrack | Multi-object tracking |
| PySide6 | Desktop GUI |
| NumPy | Numerical and data processing |
| PyInstaller | Windows EXE packaging |

---

# Development Phases

---

# Phase 1 — Project Foundation

## Title
Project Setup and Initial Computer Vision Pipeline

## Objective
Establish the basic project structure and create the initial video-processing pipeline.

## Implemented
- Python project structure
- Virtual environment
- OpenCV integration
- Basic video input handling
- Initial YOLO integration
- Initial project documentation

## Status
✅ Completed

---

# Phase 2 — Object Detection

## Title
YOLO-Based Object Detection

## Objective
Implement real-time object detection for people and vehicles using YOLO.

## Implemented
- YOLO model integration
- Object detection pipeline
- COCO object classes
- Detection filtering
- Bounding-box visualization
- Real-time webcam processing

## Supported Classes

| Class | ID |
|---|---:|
| Person | 0 |
| Bicycle | 1 |
| Car | 2 |
| Motorcycle | 3 |
| Bus | 5 |
| Truck | 7 |

## Status
✅ Completed

---

# Phase 3 — Object Tracking

## Title
Persistent Object Tracking with ByteTrack

## Objective
Extend object detection into multi-object tracking so detected objects maintain tracking IDs across frames.

## Implemented
- ByteTrack integration
- Persistent tracking IDs
- Detection + tracking pipeline
- Tracking visualization
- Multiple-object tracking

## Status
✅ Completed

---

# Phase 4 — Video Analytics

## Title
Real-Time Object Analytics

## Objective
Transform object tracking data into meaningful video analytics.

## Implemented

### Object Counting
Counts detected objects by class.

### Line Crossing
Tracks objects crossing a predefined horizontal line and records:
- Entry count
- Exit count

### Zone Analytics
Tracks objects inside a predefined rectangular zone and records:
- Current zone occupancy
- Zone entries
- Zone exits

### Dwell Time
Measures how long tracked objects remain visible.

### Trajectory Tracking
Maintains recent movement history for tracked objects and displays their trajectories.

## Status
✅ Completed

---

# Phase 5 — Desktop Application

## Title
PySide6-Based Video Analytics Interface

## Objective
Convert the computer-vision pipeline into a usable desktop application.

## Implemented
- PySide6 GUI
- Video display
- Live webcam mode
- Video-file processing
- Playback controls
- Play / pause
- Seek / timeline control
- Stop and restart
- Analytics dashboard
- Object statistics
- Line-crossing statistics
- Zone statistics
- Dwell-time statistics
- Trajectory visualization
- Dark interface design

## Status
✅ Completed

---

# Phase 6 — Testing and Cleanup

## Title
System Validation and Code Cleanup

## Objective
Validate the complete application and remove obsolete development files.

## Testing Completed

1. Application startup
2. Video detection and tracking
3. Live camera processing
4. Playback controls
5. Analytics verification
6. Stop and restart
7. Different video / edge-case testing
8. Final terminal verification

## Cleanup Completed
- Removed obsolete test scripts
- Removed unused camera module
- Removed temporary backup files
- Removed Python cache files
- Retained only active application source files
- Verified final project structure

## Status
✅ Completed

---

# Phase 7 — Documentation and Finalization

## Title

Final Documentation and Project Release

## Objective

Prepare the project for final presentation, portfolio use, GitHub, and future development.

## Final Project Structure

ai-video-analytics/
│
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── tracker.py
│   │
│   ├── analytics/
│   │   ├── counter.py
│   │   ├── dwell_time.py
│   │   ├── line_crossing.py
│   │   ├── trajectory.py
│   │   ├── zone.py
│   │   └── __init__.py
│   │
│   └── gui/
│       ├── main_window.py
│       ├── video_worker.py
│       └── __init__.py
│
├── .gitignore
├── requirements.txt
├── README.md
├── AI-Video-Analytics.iss
└── yolo11n.pt


---

# Phase 8 — Windows EXE Packaging

The application was packaged as a standalone Windows desktop executable using PyInstaller.

## Packaging

The final executable was built using:

python -m PyInstaller --noconfirm --clean --windowed --name AI-Video-Analytics app\main.py


The final PyInstaller application was successfully tested on Windows.

The packaged application includes the required Python runtime, application dependencies, YOLO model, and supporting files.

---

# Phase 9 — Final Release

The final Windows release was packaged and tested successfully.

## Portable Release

The portable application is available in:

release/
└── AI-Video-Analytics/
    ├── AI-Video-Analytics.exe
    ├── yolo11n.pt
    ├── README.md
    └── _internal/

The portable version can be run directly using:

AI-Video-Analytics.exe

## Windows Installer

A proper Windows installer was created using **Inno Setup**.

Installer:

release/
└── AI-Video-Analytics-Setup.exe

The installer:

* Installs the complete application on Windows.
* Includes the YOLO model.
* Includes all PyInstaller runtime dependencies.
* Creates Start Menu and Desktop shortcuts.
* Provides an uninstall option through Windows.
* Was installed and tested successfully.

The installer configuration is maintained in:

AI-Video-Analytics.iss

## Final Release Verification

The following were successfully verified:

* Portable release launches correctly.
* Windows installer completes successfully.
* Installed application launches correctly.
* Camera/video processing works correctly.
* YOLO object detection works correctly.
* Object tracking works correctly.
* Video analytics work correctly.
* Required PyTorch/TorchVision compatibility is included.
* Final Git working tree is kept free of generated build artifacts.

## Final Distribution

For distributing the application to another Windows computer, use:

release\AI-Video-Analytics-Setup.exe

The portable release folder is retained as an alternative distribution and backup package.

