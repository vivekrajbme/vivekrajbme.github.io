---
layout: single
title: "Research"
permalink: /research/
author_profile: true
author: vivek-raj
---

## The Problem

Myoelectric prostheses control hand and wrist movement using EMG electrodes on the skin. This requires direct contact, frequent recalibration, and expert fitting — contributing to the roughly 50% abandonment rate of myoelectric prostheses worldwide.

## The Approach: Optical Myography (OMG)

My PhD work replaces EMG electrodes with a single standard USB camera. The camera observes the residual limb and decodes intended grasp, wrist flexion-extension, and pronation-supination — contactless, with no calibration burden.

I developed this across three progressively less-constrained pipelines:

**1. Marker-assisted control** — Reflective markers on the forearm are tracked with a Kalman-filtered computer vision pipeline; angular features drive a regression model for proportional control. This established the feasibility and accuracy ceiling of the approach.

**2. Markerless control (classical CV)** — Removes the need for markers entirely, using skin-tone and shape-based segmentation to track the residual limb directly, extending the system to work for individuals with amputation without any attached hardware.

**3. Markerless control (deep learning)** — A lightweight neural segmentation model (trained using Meta AI's SAM2 to auto-generate labels, since no public dataset of residual limbs exists) makes the system robust enough for real-world deployment on embedded hardware.

## Results

| Measure | Outcome |
|---|---|
| End-to-end latency | < 90 ms — real-time control at 11–30 Hz |
| Able-bodied task success (n=13) | 90.6–100%, 0.53–0.56 bits/s throughput |
| Task success, individuals with transradial amputation (n=3, AIIMS-approved trial) | 83–100%, 0.29–0.47 bits/s throughput |

This is the first real-time, closed-loop, vision-based proportional prosthetic control demonstrated with individuals with amputation in India, matching published high-density EMG benchmarks at under 5% of the hardware cost.

## What This Solves

A cheaper, contactless, easier-to-fit alternative to EMG-based prosthetic control — validated end-to-end from lab bench to clinical trial, with a filed patent and a growing publication record.

---

## Publications & Patent

- **[Published]** *Toward Markerless, Noncontact Optical Myography for Prosthetic Control of Multiple Degrees of Freedom*, IEEE Sensors Journal, Feb 2025. [DOI: 10.1109/JSEN.2024.3512454](https://ieeexplore.ieee.org/abstract/document/10811820)
- **[Under Review]** *Optical Myography-based Measurement of Proportional Hand and Wrist Kinematics for Prosthetic Control*, JNER
- **[Under Review]** *A Single-Camera Reflective-Marker System for Real-Time Proportional Control of Wrist and Hand Motions*, IEEE Sensors Journal
- **[Filed, Nov 2024]** Patent — Optical Myography-based motion intent detection for prosthetic control, FITT, IIT Delhi
- *Feature Selection for Attention Demanding Task Induced EEG Detection*, IEEE ASPCON, 2020. [DOI: 10.1109/ASPCON49795.2020.9276710](https://ieeexplore.ieee.org/abstract/document/9276710)
