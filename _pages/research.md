---
layout: single
title: "Research"
permalink: /research/
author_profile: true
author: vivek-raj
---

## 📝 Overview

My PhD research develops **real-time Optical Myography (OMG)** — a camera-based alternative to EMG — for **proportional control of upper-limb prostheses**. Across three progressively less-constrained pipelines (marker-assisted → markerless classical → markerless deep learning), I've built and clinically validated a system that:

- **Tracks** hand/forearm motion from a single USB camera, with and without reflective markers  
- **Fuses** vision, IMU, and data-glove signals for ground truth and orientation estimation  
- **Classifies/regresses** grasp, wrist flexion-extension, and pronation-supination in real time  
- **Controls** a proportional prosthetic interface at <90 ms end-to-end latency  
- Has been **clinically validated on transradial amputees at AIIMS New Delhi** (AIIMS IEC A00086/03.11.2023), not just able-bodied subjects

---

## 🔬 Methods

### 1. Marker-Based Visual Tracking  
- **Preprocessing**: Applied morphological opening/closing and area thresholds to isolate high-speed reflective markers on the forearm.  
- **Filtering**: Implemented a constant-velocity Kalman filter with adaptive process noise to handle marker occlusions and intermittent disappearances.  
- **Performance**: Achieved an average detection rate of **14 out of 16 markers**, improving over baseline methods.

### 2. IMU Orientation Estimation  
- **Hardware**: Integrated an Adafruit BNO08x IMU (9-DOF) with a QT Py microcontroller, and earlier prototypes using FXOS8700 + FXAS21002 on a Teensy LC.  
- **Sensor Fusion**: Utilized onboard fusion algorithms to compute quaternions, then extracted yaw, pitch, and roll in MATLAB.  
- **Accuracy**: Maintained orientation RMSE below **2°** across all three Euler angles.

### 3. Multimodal Data Synchronization  
- **Setup**: Synchronized video frames (`snapshot(webcam)`) with IMU logs. Prompts were rendered on an external display to cue subjects.  
- **Implementation**: Streamlined RealTimeDC MATLAB code to capture imagery and sensor streams in a unified loop, ensuring <50 ms end-to-end latency.

### 4. Intent Classification & Model Training  
- **Feature Extraction**: Segmented data around movement peaks, extracting time-domain features (e.g., mean, variance, peak amplitudes).  
- **Models**: Evaluated Ridge Regression (RR), Support Vector Regression (SVR), CatBoost, and LSTM networks.  
- **Validation**: Employed peak-based k-fold cross-validation to avoid data leakage, using MSE, NRMSE, and R² as metrics.  
- **Results**: LSTM achieved **93% classification accuracy** on six grasp/pronation-supination gestures.

### 5. Markerless Control for Amputees (Pipeline B)
- **StumpSegmenter**: Otsu thresholding + YCrCb skin-tone masking for residual-limb segmentation — no markers, no electrodes, no contact.  
- **Feature selection**: shape, texture, and pose features selected by `|Pearson r| > 0.3` and motion SNR > 2.0, with EMA smoothing (α = 0.85–0.92) for jitter suppression.  
- **Results**: MSE 0.031 (Linear) / 0.063 (SVR) offline; extended to a validated real-time TAC loop across multiple amputee subjects.

### 6. Markerless Control via Deep Learning (Pipeline C)
- **Model**: MobileNetV3-small encoder + custom depthwise decoder, trained on **SAM2 (Meta AI)** pseudo-labels (IoU > 0.9) — solving the lack of any public amputee OMG dataset.  
- **Training**: Combined BCE + Dice loss; CPU-inference validated for embedded deployment; exported to **ONNX** for cross-platform use.  
- **Results**: 100% TAC success rate in able-bodied preliminary validation.

---

## 📊 Results & Impact

| Component               | Metric / Outcome                                   |
|--------------------------|-----------------------------------------------------|
| Marker Tracking          | 87.5% marker detection rate (14/16)                 |
| Orientation Estimation   | < 2° RMSE (yaw, pitch, roll)                        |
| Gesture Classification   | 93% accuracy on 6 gestures (LSTM)                   |
| End-to-end Latency       | < 90 ms, real-time control loop at 11–30 Hz         |
| Able-bodied TAC (n=13)   | 90.6–100% task success, 0.53–0.56 bits/s throughput |
| Transradial amputees (n=3, AIIMS) | 83–100% task success, 0.29–0.47 bits/s throughput |

These results — validated on **both able-bodied and amputee subjects in a clinical setting** — demonstrate a feasible, real-time, contactless motion-intent decoding system that matches published high-density EMG benchmarks at under 5% of the hardware cost.

---

## 🚀 Future Directions

- **Embedded Deployment**: Port inference to edge GPUs (e.g., Jetson Nano) for fully untethered, on-device operation.  
- **Multimodal Fusion**: Integrate EMG signals alongside vision for richer intent cues where contact is acceptable.  
- **Adaptive Learning**: Online calibration methods to personalize models per user without retraining from scratch.  
- **Broader Clinical Validation**: Expand the amputee cohort and long-term at-home evaluation with rehabilitation partners.

---

*This body of work — 1 published journal paper, 2 under review, and a filed patent — lays the foundation for next-generation, contactless human-machine interfaces that can restore upper-limb function without the cost and calibration burden of EMG.*
