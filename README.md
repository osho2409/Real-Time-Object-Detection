# Real-Time Object Detection, Tracking and Counting using YOLO11n

## Overview

This repository presents a complete edge AI pipeline for real-time object detection, tracking, counting, and mobile deployment using **YOLO11n**.

The project was developed to detect three everyday object classes:

* Water Bottle
* Mug
* Remote

The implementation is divided into two major components:

* **Desktop Implementation** – Real-time detection, object tracking, and unique object counting using the trained YOLO11n PyTorch model (`best.pt`).
* **Android Implementation** – On-device real-time object detection using the exported YOLO11n ONNX model (`best.onnx`) integrated with Flutter and ONNX Runtime.

The project demonstrates the complete workflow from dataset preparation and model training to cross-platform deployment on desktop and Android devices.

---

# Repository Structure

```text
Real-Time-Object-Detection/
│
├── desktop/
│   ├── model/
│   ├── trackers/
│   ├── notebooks/
│   ├── dataset/
│   └── requirements.txt
│
├── mobile/
│   ├── android/
│   ├── assets/
│   ├── lib/
│   ├── test/
│   ├── analysis_options.yaml
│   └── pubspec.yaml
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# Project Pipeline

```
Dataset Collection
        │
        ▼
Dataset Annotation
        │
        ▼
YOLO11n Fine-Tuning
        │
        ▼
Desktop Detection & Tracking (.pt)
        │
        ▼
ONNX Export
        │
        ▼
Android Deployment (.onnx)
        │
        ▼
Real-Time On-Device Inference
```

---

# Dataset

The dataset consists of a combination of:

* Open-source annotated images
* Personally collected images
* Manually annotated images

The final dataset contains three object classes:

* Bottle
* Mug
* Remote

The remote class was created through manual image collection and annotation to extend the dataset beyond the available open-source classes.

---

# Why YOLO11n?

YOLO11n was selected because it provides an effective balance between inference speed and detection accuracy for edge devices.

Compared to larger YOLO variants, YOLO11n offers:

* Lower computational complexity
* Smaller model size
* Faster inference
* Lower memory consumption
* Better suitability for CPU-only execution
* Practical deployment on mobile hardware

These characteristics make YOLO11n well suited for real-time applications where latency is more critical than maximizing absolute detection accuracy.

---

# Desktop Implementation

The desktop implementation uses the trained **YOLO11n PyTorch model (****`best.pt`****)**.

Features include:

* Real-time object detection
* Object tracking
* Unique object counting
* Live visualization
* Per-class object statistics

The final implementation is contained in:

```
desktop/trackers/custom_tracker.py
```

---

# Tracking Approaches Explored

Several tracking methods were investigated during development before selecting the final implementation.

These files are retained for documentation purposes:

* tracker.py
* histogram_based_tracker.py
* resnet_tracker.py
* deepsort_test.py

These implementations represent experimental approaches evaluated during development and are **not part of the final desktop pipeline**. Only custom_tracker.py is the tracker that is built for this specific case and used in the implementation.

---

# Mobile Implementation

The Android application deploys the same trained YOLO11n model after exporting it to the ONNX format.

The mobile application is built using:

* Flutter
* ONNX Runtime
* Android Camera API

The mobile application performs:

* Real-time camera inference
* On-device object detection
* Live bounding box visualization
* FPS measurement

Unlike the desktop implementation, the Android version focuses on lightweight real-time inference and hardware benchmarking.

---

# Model Formats

Two model formats are used in this project depending on the deployment environment.

### Desktop

```
best.pt
```

Used directly with the Ultralytics YOLO framework for detection, tracking, and counting.

### Android

```
best.onnx
```

Exported from the trained YOLO11n model and deployed using ONNX Runtime for efficient on-device inference.

Although the formats differ, both originate from the same trained YOLO11n model.

---

# Performance Optimization

Several optimizations were applied to improve inference performance on Android.

## Reduced Input Resolution

The original training resolution was reduced during deployment.

Input image size:

```
320 × 320
```

Reducing the input resolution significantly improves inference speed while maintaining acceptable detection accuracy.

---

## NNAPI Acceleration

Android Neural Networks API (NNAPI) was enabled to utilize available hardware acceleration whenever supported by the device.

---

## Graph Optimizations

ONNX Runtime graph optimizations were enabled to reduce inference latency.

---

## Multi-threading

Inference thread configuration was optimized to reduce CPU overhead.

---

## Frame Skipping

Frames are selectively skipped when required to maintain smoother application performance under heavy workloads.

---

# Engineering Trade-offs

Several design decisions were made throughout development.

| Decision                    | Benefit                               | Trade-off                                                 |
| --------------------------- | ------------------------------------- | --------------------------------------------------------- |
| YOLO11n                     | Lightweight and fast                  | Slightly lower accuracy than larger YOLO variants         |
| 320×320 input               | Higher FPS                            | Reduced small-object detail                               |
| ONNX Runtime                | Cross-platform deployment             | Additional model conversion step                          |
| NNAPI acceleration          | Faster inference on supported devices | Performance depends on hardware support                   |
| Lightweight mobile pipeline | Stable real-time inference            | Tracking and counting omitted from Android implementation |

---

# Technologies Used

## Machine Learning

* YOLO11n
* PyTorch
* Ultralytics

## Computer Vision

* OpenCV
* NumPy

## Model Deployment

* ONNX
* ONNX Runtime

## Mobile Development

* Flutter
* Dart
* Android Camera API

---

# Desktop Usage

Install dependencies

```bash
pip install -r requirements.txt
```

Run the desktop application

```bash
python desktop/trackers/custom_tracker.py
```

---

# Android Usage

Navigate to the mobile project.

Install Flutter dependencies.

```bash
flutter pub get
```

Run the application.

```bash
flutter run
```

---

# Experimental Work

Several implementation strategies were explored throughout the project, including:

* Histogram-based tracking
* Appearance-based tracking
* DeepSORT evaluation
* ResNet-based feature extraction
* Alternative ONNX export methods
* Multiple deployment optimizations

These experiments are included to document the development process and engineering decisions that led to the final implementation.

---

# Current Limitations

The system may experience reduced performance under the following conditions:

* Heavy object occlusion
* Motion blur
* Very small objects
* Extremely low-light environments
* Multiple overlapping objects
* Uncommon object orientations
* Long horizontal narrow objects gets confused for remotes and bottles.

During testing, however, the deployed Android application demonstrated successful detection of some horizontally oriented bottles that were previously challenging during desktop experimentation.

---

# Future Improvements

Potential extensions include:

* Multi-camera support
* Additional object classes
* INT8 quantization
* Appearance embedding for mobile tracking
* Cross-platform deployment on embedded edge devices
* Real-time analytics dashboard

---

# Acknowledgements

This project utilizes:

* Ultralytics YOLO11
* ONNX Runtime
* Flutter
* OpenCV
* PyTorch

along with publicly available datasets and manually annotated images collected specifically for this project.

---

# License

This repository is intended for academic, educational, and research purposes.
