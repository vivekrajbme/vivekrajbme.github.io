---
layout: single
title: "CV Stage 5: Specialization & Interview Prep"
permalink: /learn-computer-vision/stage-5-specialization/
author_profile: true
author: vivek-raj
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: "3D vision, video, efficient deployment, medical imaging, SLAM — and how to prepare for the interview itself."
---

[← Back to the full curriculum](/learn-computer-vision/) · [← Stage 4: Modern Era](/learn-computer-vision/stage-4-modern-era/)

Once you're through Stage 4, depth beats breadth. Pick one track below and go deep enough to defend it under follow-up questions — that's what separates a senior offer from a generalist screen pass.

---

## 5.1 3D Vision: NeRF and Gaussian Splatting

**NeRF** (Mildenhall et al., 2020) represents a scene as a continuous volumetric function — a small MLP that maps a 3D position + viewing direction to color and density — queried along camera rays and integrated to render a pixel.

```python
import torch
import torch.nn as nn

class TinyNeRF(nn.Module):
    def __init__(self, pos_dim=60, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),   # RGB + density (sigma)
        )

    def forward(self, encoded_position):
        out = self.net(encoded_position)
        rgb, sigma = torch.sigmoid(out[..., :3]), torch.relu(out[..., 3])
        return rgb, sigma

def volume_render(rgb, sigma, deltas):
    """Classic volume rendering integral, discretized along a ray."""
    alpha = 1 - torch.exp(-sigma * deltas)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha + 1e-10]), dim=0)[:-1]
    weights = transmittance * alpha
    return (weights[..., None] * rgb).sum(dim=0)
```

**3D Gaussian Splatting** (Kerbl et al., 2023) trades NeRF's implicit MLP for an explicit set of 3D Gaussians (position, covariance, color, opacity) that are directly rasterized rather than ray-marched — this is why it renders in real time where NeRF does not, and it's now the preferred approach for novel-view synthesis in production.

---

## 5.2 Video Understanding

3D CNNs (C3D, I3D — convolve over space *and* time) → two-stream networks (separate RGB and optical-flow streams, fused late) → video transformers (TimeSformer, VideoMAE — apply attention over space-time patches).

```python
import torch.nn as nn

class Simple3DConv(nn.Module):
    """A 3D conv kernel slides over (time, height, width) simultaneously."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):  # x: (B, C, T, H, W)
        return self.conv3d(x)
```

**The tradeoff to be able to state clearly:** 3D convolution captures short-range motion cheaply but scales poorly to long clips; video transformers scale better to long-range temporal reasoning but need either factorized (space then time) attention or heavy compute to stay tractable — that factorization is exactly what TimeSformer proposes.

---

## 5.3 Efficient / Edge Deployment

The default two-stage recipe in industry now: train (or fine-tune) a large model, then compress it for whatever it actually has to run on.

```python
import torch

# Post-training dynamic quantization: FP32 weights -> INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Structured pruning: remove entire channels, not just individual weights
import torch.nn.utils.prune as prune
prune.ln_structured(model.conv1, name="weight", amount=0.3, n=2, dim=0)

# Knowledge distillation: train a small "student" to match a large "teacher"
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

MobileNet/ShuffleNet-style **depthwise-separable convolutions** split a standard convolution into a per-channel spatial conv plus a 1×1 conv across channels — roughly an 8–9× reduction in multiply-adds for a typical 3×3 kernel, which is *why* they exist, not just a fun fact.

This isn't academic for me — my own segmentation model has to run in real time (under 90ms end-to-end) on embedded hardware for prosthetic control. Distillation and quantization aren't a side chapter there; they're the difference between a system that works on a patient's arm and one that doesn't.

---

## 5.4 Medical / Biomedical Imaging

The core difference from general-purpose CV: **small-data regimes** (labeled medical data is expensive and often requires clinical expertise to annotate), **domain shift** (a model trained on one hospital's scanner often degrades on another's), and a different validation bar — sensitivity/specificity and clinical trial evidence matter more than top-1 accuracy, and deployment typically requires ethics/IRB approval. U-Net and its many variants remain the workhorse architecture here precisely because it performs well in low-data regimes.

## 5.5 Robotics / SLAM

Visual-inertial odometry fuses camera and IMU data; **loop closure** recognizes a previously visited place to correct accumulated drift; modern systems combine classical geometry (Stage 2's epipolar constraints) with learned feature matching (SuperPoint for detection, SuperGlue for matching) for far more robust correspondences than classical descriptors alone provide in textureless or repetitive environments.

---

## Current State of the Art (2025–2026 Snapshot)

- **Segmentation:** SAM2 and its derivatives dominate promptable segmentation, including video, near real-time.
- **Representation learning:** DINOv2-class self-supervised backbones are now standard "foundation" vision encoders, often outperforming supervised pretraining for downstream transfer.
- **Multimodal reasoning:** frontier multimodal LLMs handle document understanding, chart/diagram reasoning, and grounded visual QA approaching specialist-model performance.
- **Generation:** diffusion-based image and video models keep improving temporal consistency and controllability; latent-space diffusion remains the dominant recipe for compute efficiency.
- **Efficient deployment:** "train or distill a huge model, then compress it" is now the default two-stage recipe for anything shipping to a device.
- **3D:** Gaussian Splatting has largely displaced vanilla NeRF for real-time applications, with active research on dynamic (4D) scenes.

The throughline: **general-purpose foundation models pretrained at scale, then adapted cheaply**, has replaced "train a bespoke model per task." Understanding *why* that shift happened — pretraining amortizes the expensive part of learning general visual structure across every downstream task — tells you where the field is heading, not just where it's been.

---

## How Companies Actually Interview for This

Four buckets, from both sides of the table:

1. **Fundamentals, derived not recalled** — "Why is convolution equivariant but pooling isn't?" "Derive the receptive field of a 3-layer 3×3 conv stack." "Why does batch norm fail at batch size 1?" They're checking mechanism, not vocabulary.
2. **Coding, usually classical CV or basic tensor ops** — implement NMS, IoU, a sliding-window convolution, or connected-component labeling from scratch in plain NumPy/Python, no framework crutch. (Every one of these is written out, working, in Stages 1–3 above — go re-derive them without looking.)
3. **System design for vision** — "Design a real-time face recognition system for a building entrance," or "design a defect-detection pipeline for a manufacturing line." They want the full pipeline: data collection/labeling cost, model choice given latency/accuracy/hardware constraints, class imbalance and distribution shift, monitoring after deployment — not "use ResNet."
4. **Paper discussion** — pick 2–3 papers you can discuss deeper than the abstract: what problem existed before it, what specifically it changed, what its limitation was, and what paper fixed *that*. Pick one per stage above rather than trying to cover everything shallowly.

**Honest advice, from both sides of this table:** breadth gets you through the resume screen; depth on a handful of topics you can defend under follow-up questions is what gets you the offer.

---

## Master Resource List

**Books**
- Szeliski, *Computer Vision: Algorithms and Applications* — free at [szeliski.org/Book](https://szeliski.org/Book/).
- Goodfellow, Bengio & Courville, *Deep Learning* — free at [deeplearningbook.org](https://www.deeplearningbook.org/).
- Bishop, *Pattern Recognition and Machine Learning* — the probability/Bayesian backbone behind half of this curriculum.

**Courses**
- [Stanford CS231n](https://cs231n.stanford.edu/) — the canonical path through Stages 1–3.
- [fast.ai](https://course.fast.ai/) — fastest way to get hands-on and training real models.
- [Hugging Face courses](https://huggingface.co/learn) — Diffusion Models course, Deep RL course, and the general Transformers course, all directly relevant to Stage 4.

**People / channels to follow**
- Andrej Karpathy — [@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy) — from-scratch build videos.
- Yannic Kilcher — [@YannicKilcher](https://www.youtube.com/@YannicKilcher) — deep paper walkthroughs.
- Two Minute Papers — [@TwoMinutePapers](https://www.youtube.com/@TwoMinutePapers) — fast visual intuition on new results.

**Sites**
- [Papers with Code](https://paperswithcode.com/) — leaderboards + reference implementations.
- [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) — the field's actual pulse; skim titles weekly even before you can read every paper deeply.
- [PyImageSearch](https://pyimagesearch.com/) — practical, code-first tutorials, strongest for Stages 1–2.

---

*Have a specific paper you want unpacked further, or want to work through a mock system-design interview question together? [Reach out](/contact/) — I enjoy these conversations more than almost anything else about this field.*
