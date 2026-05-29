# Real-time-Multi-Stream-Image-and-Hybrid-Control-for-Self-Driving-Capabilities

## Overview

This project is a real-time autonomous driving assistance system that combines:

* Classical Digital Image Processing (DIP)
* YOLOv8 Object Detection
* Multi-stream camera processing
* Hybrid navigation and control logic

The system performs:

* Lane detection
* Obstacle detection
* Driving command generation
* Multi-camera monitoring

It is designed to run efficiently on standard CPU-based systems without requiring a dedicated GPU.

---

## Features

### Lane Detection

* Canny Edge Detection
* Hough Line Transform
* K-Means Road Segmentation
* ROI Masking
* Polynomial Lane Fitting
* Temporal Lane Smoothing
* Ghost Mode for temporary lane loss

### Object Detection

* YOLOv8 Nano model
* Real-time obstacle detection
* Threaded YOLO inference
* Proximity classification
* Green lane-mask filtering

### Navigation Commands

The system generates:

* FORWARD
* STOP
* TURN_LEFT
* TURN_RIGHT
* REVERSE

### Multi-Stream Support

* Front camera processing
* Rear camera obstacle monitoring
* Reverse decision logic
* Parallel stream handling using Python threading

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Ultralytics YOLOv8
* Threading
* Queue
* Digital Image Processing techniques

---

## Project Structure

```bash
├── fn_lanedetection.py    # Lane detection engine
├── single_stream.py       # Optimized single camera pipeline
├── dual_camera.py         # Front + rear camera processing
├── yolo_detect.py         # YOLO detection and command logic
├── yolo_metrics.py        # Metrics and evaluation engine
├── datasets/              # Test videos and datasets
└── outputs/               # Generated outputs and results
```

---

## System Workflow

1. Capture video frames
2. Perform lane detection
3. Run YOLO object detection
4. Classify obstacle proximity
5. Compute navigation command
6. Smooth commands
7. Display overlays and metrics

---



## Performance Highlights

* Real-time processing on CPU
* Average FPS: ~11–12 FPS
* Stable lane tracking
* Multi-stream support
* Optimized threaded YOLO inference

---

## Dataset

The project was tested on:

* Daylight roads
* Nighttime roads
* Curved roads
* Roads without markings
* Multi-camera driving scenarios

Objects detected include:

* Cars
* Persons
* Motorcycles
* Plants
* Road obstacles

---

## Future Improvements

* GPU acceleration
* Curved lane fitting using polynomial degree-2
* Depth estimation
* Steering actuator integration
* Real vehicle deployment

---

## Authors

* Rida Zahra
* Tazeen ur Rehman
* Mawa Chaudhary
---

## Output:

https://nustedupk0-my.sharepoint.com/personal/mchaudry_ce45ceme_student_nust_edu_pk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmchaudry%5Fce45ceme%5Fstudent%5Fnust%5Fedu%5Fpk%2FDocuments%2FOutput%5FDIP%5FProject&viewid=7d922e70%2D0927%2D4e6f%2D9eea%2D67e38f48e1c8
