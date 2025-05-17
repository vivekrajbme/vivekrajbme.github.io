---
layout: single
title: "Research"
permalink: /research/
author_profile: true
author: vivek-raj
---

## 📝 Overview

My PhD research develops **real-time algorithms** for decoding upper-limb motion intent to drive **assistive and rehabilitation devices**. By combining **computer-vision marker tracking**, **IMU-based orientation estimation**, and **machine-learning classification**, I’ve built a robust pipeline that:

- **Tracks** reflective markers on the forearm with high accuracy  
- **Fuses** vision and inertial data to estimate joint angles  
- **Classifies** intended grasp and pronation–supination gestures  
- **Controls** a proportional assistive interface in real time  

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

---

## 📊 Results & Impact

| Component             | Metric / Outcome                         |
|-----------------------|------------------------------------------|
| Marker Tracking       | 87.5% marker detection rate (14/16)      |
| Orientation Estimation| < 2° RMSE (yaw, pitch, roll)             |
| Classification        | 93% accuracy on 6 gestures (LSTM)        |
| Latency               | < 50 ms total loop time                  |

These results demonstrate a **feasible, real-time motion-intent decoding system** suitable for embedded assistive devices and rehabilitation robotics.

---

## 🚀 Future Directions

- **Embedded Deployment**: Port the LSTM inference to edge GPUs (e.g., Jetson Nano) for untethered operation.  
- **Multimodal Fusion**: Integrate EMG signals alongside vision and IMU for richer intent cues.  
- **Adaptive Learning**: Develop online calibration methods to personalize models per user.  
- **Clinical Validation**: Collaborate with rehabilitation practitioners to evaluate efficacy in patient trials.

---

*This body of work lays the foundation for next-generation human-machine interfaces that can restore or augment upper-limb function with high accuracy and low latency.*  
