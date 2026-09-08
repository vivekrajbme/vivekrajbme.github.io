---
layout: single
title: "Learn Computer Vision: Beginner to Expert"
permalink: /learn-computer-vision/
author_profile: true
author: vivek-raj
toc: true
toc_label: "Curriculum"
toc_icon: "eye"
toc_sticky: true
excerpt: "A mentor's roadmap through computer vision — theory, papers, and what interviewers actually ask."
---

*Notes from the mentor sessions I wish I'd had when I started my PhD. Written the way I'd explain it on a whiteboard: build the intuition first, then hang the math and the papers off it.*

Think of computer vision as answering one question over and over, at increasing levels of sophistication: **"What can I recover about the 3D world from a 2D grid of numbers?"** Every topic below — from edge detection to diffusion models — is a different tool for answering that question. Keep that thread and the field stops looking like a pile of disconnected tricks.

The roadmap has five stages. Don't skip stage 1 to get to the exciting deep learning parts — I have interviewed candidates who could recite ResNet's architecture but couldn't explain why a Sobel filter is separable. That gap gets found in the first interview round.

**This page is the syllabus.** Each stage below now has its own deep-dive lecture page with full theoretical explanations, working code you can run, exercises, and curated links to papers, courses, and videos:

1. [Stage 1 — Foundations](/learn-computer-vision/stage-1-foundations/): image formation, sampling, linear algebra, classical image processing (built from scratch in NumPy)
2. [Stage 2 — Classical Computer Vision](/learn-computer-vision/stage-2-classical-cv/): feature detection, geometric vision, pre-deep-learning segmentation
3. [Stage 3 — Deep Learning for Vision](/learn-computer-vision/stage-3-deep-learning/): CNNs, training mechanics, object detection, segmentation
4. [Stage 4 — Transformers, Self-Supervision & Generative Models](/learn-computer-vision/stage-4-modern-era/): ViT, SimCLR/BYOL/DINO, CLIP/LLaVA, GANs/diffusion
5. [Stage 5 — Specialization & Interview Prep](/learn-computer-vision/stage-5-specialization/): NeRF/Gaussian Splatting, video, edge deployment, current SOTA, and how CV interviews are actually structured

Work through them in order — each one assumes the last. What follows below is the same roadmap in overview form; use it as a map back to any topic.

---

## Stage 1 — Foundations (Weeks 1–4)

### 1.1 How an image is formed

A digital image is a discretized, quantized sample of the *plenoptic function* — light intensity as a function of position, direction, wavelength, and time. A camera collapses this down to a 2D array by:

1. **Perspective projection** — the pinhole camera model maps a 3D point $(X, Y, Z)$ to image coordinates via $x = fX/Z$, $y = fY/Z$. Everything in multi-view geometry (Stage 4) is a variation on this equation.
2. **Sampling** — the continuous image is sampled on a pixel grid. Undersample and you get **aliasing** (the reason checkerboard patterns shimmer in photos and why CNNs need anti-aliased downsampling — see Zhang's *"Making Convolutional Networks Shift-Invariant Again"*, ICML 2019).
3. **Quantization** — continuous intensity is rounded to discrete levels (8-bit, 16-bit, HDR).

**Why this matters for interviews:** "Why does bilinear interpolation blur an upsampled image?" and "why do you get moiré patterns when you photograph a monitor?" are both direct consequences of sampling theory (Nyquist-Shannon). If you can derive the answer instead of remembering it, you separate yourself immediately.

### 1.2 Linear algebra and probability you actually need

Don't relearn all of linear algebra — learn these with a CV lens:

- **Eigen-decomposition / SVD** → PCA (eigenfaces), whitening, low-rank approximation of weight matrices (used in modern model compression).
- **Convolution as matrix multiplication** (im2col) → why GPUs are fast at CNNs, and why grouped/depthwise convolutions save FLOPs.
- **Homogeneous coordinates** → every geometric transform (rotation, translation, projection) becomes a single matrix multiply. This is the language of Stage 4.
- **Bayes' rule and MAP estimation** → the mathematical spine of denoising, Kalman filters, and (later) diffusion models — a diffusion model is literally learning the score of $\nabla \log p(x)$.

### 1.3 Classical image processing

Build these from scratch at least once — with NumPy, not OpenCV — before you ever call a library function:

| Concept | Core idea | Where it resurfaces later |
|---|---|---|
| Convolution / correlation | Sliding weighted sum | The literal operation inside every CNN layer |
| Gaussian & separable filters | Blur is a low-pass filter; separable = $O(n)$ not $O(n^2)$ | Feature pyramids, anti-aliasing |
| Sobel / Canny edge detection | Gradient magnitude + non-max suppression + hysteresis | Intuition for what early CNN filters learn |
| Histogram equalization | Redistribute intensities for contrast | Data normalization philosophy |
| Morphological ops (erosion/dilation) | Shape-based filtering on binary masks | Post-processing segmentation masks |
| Fourier transform of images | Frequency-domain view of an image | Explains JPEG compression, and why CNNs have a "texture bias" (Geirhos et al., ICLR 2019) |

**Exercise I give every mentee:** implement Canny edge detection from scratch, then explain why each of its three stages (Gaussian smoothing, gradient computation, non-max suppression + double threshold) exists. If you can't answer "why not just threshold the gradient magnitude directly," you don't understand hysteresis yet.

---

## Stage 2 — Classical Computer Vision (Weeks 5–10)

This stage still gets asked about in interviews at companies doing AR/VR, robotics, or camera pipelines — deep learning didn't delete this knowledge, it moved it into "explain what a CNN is implicitly learning."

### 2.1 Feature detection and description

- **Harris corners** — a corner is a point where the local autocorrelation changes in *all* directions (eigenvalues of the structure tensor are both large).
- **SIFT** (Lowe, 2004) — scale-space extrema via Difference-of-Gaussians, orientation histograms for rotation invariance. This single paper (35,000+ citations) defined "hand-crafted feature descriptor" as a category and powered a decade of panorama stitching, SLAM, and image retrieval.
- **HOG** (Dalal & Triggs, CVPR 2005) — gradient orientation histograms over cells; the feature behind the original real-time pedestrian detectors.
- **ORB / FAST / BRIEF** — binary descriptors built for speed on embedded devices; still used today in visual-inertial odometry on drones and phones because they're cheap enough to run at 200Hz.

**Interview framing:** "Explain SIFT's scale-invariance" tests whether you understand *why* Gaussian scale-space plus extrema detection gives you invariance, not whether you memorized the pipeline diagram.

### 2.2 Geometric vision

- **Camera calibration** — intrinsic matrix $K$, distortion coefficients, extrinsic $[R|t]$. Zhang's checkerboard method (2000) is still the industry-standard calibration technique.
- **Epipolar geometry** — the fundamental/essential matrix constrains where a point in one image can appear in another. This is *the* concept underlying stereo depth, structure-from-motion, and visual SLAM.
- **Stereo vision & disparity** — depth is inversely proportional to disparity: $Z = fB/d$. Classical block-matching stereo (SGM) is still deployed in real products because it's deterministic and doesn't need a GPU.
- **Optical flow** — Lucas-Kanade (local, sparse) vs. Horn-Schunck (global, dense). Sets up modern flow networks (FlowNet, RAFT).

### 2.3 Segmentation and clustering (pre-deep-learning)

Watershed, GrabCut, mean-shift, superpixels (SLIC). Know these exist and their failure modes — a lot of "why did DeepLab do X" interview questions are really "what problem did classical segmentation have that this solves."

---

## Stage 3 — Deep Learning for Vision (Months 3–7)

This is the stage where most self-taught engineers start, and where most of them develop gaps because they skipped Stages 1–2. Don't be that candidate.

### 3.1 The convolutional neural network, properly understood

A convolution layer is doing three things simultaneously: **local connectivity** (each output depends on a small receptive field), **weight sharing** (same filter slides everywhere → translation equivariance), and **hierarchical composition** (stacking layers grows the receptive field and abstraction level). Interviewers love asking "why not just use a fully connected layer" — the answer is parameter count *and* the inductive bias of translation equivariance, which is a much better answer than "it's faster."

**Canonical architecture lineage** (know what each one *introduced*, not just its name):

| Year | Model | What it introduced |
|---|---|---|
| 1998 | LeNet-5 (LeCun) | Conv + pooling for digit recognition — the template |
| 2012 | AlexNet (Krizhevsky et al.) | ReLU, dropout, GPU training — ignited the deep learning era at ImageNet |
| 2014 | VGG (Simonyan & Zisserman) | Stacked 3×3 convs = larger receptive field with fewer parameters than one big filter |
| 2014 | GoogLeNet/Inception | Multi-scale filters in parallel (inception module), 1×1 convs for dimensionality reduction |
| 2015 | ResNet (He et al.) | Residual/skip connections solve vanishing gradients — arguably the single most important architectural idea in deep learning, not just vision |
| 2017 | DenseNet | Feature reuse via dense connections |
| 2019 | EfficientNet (Tan & Le) | Compound scaling (depth/width/resolution jointly) via neural architecture search |
| 2020 | Vision Transformer (Dosovitskiy et al.) | Images as patch sequences — no convolutional inductive bias at all, needs scale to work |

**The ResNet question you must be able to answer:** why do skip connections help? Because they let gradients flow directly through identity mappings during backprop, and because the network only has to learn a *residual* function $F(x) = H(x) - x$ rather than the full mapping — an easier optimization problem. This is asked in nearly every CV interview I've sat on.

### 3.2 Training mechanics that separate juniors from seniors

- **Data augmentation** — flips, crops, color jitter, Mixup, CutMix, RandAugment. Understand augmentation as a way of encoding invariances you want the model to have, not a checkbox.
- **Normalization** — BatchNorm vs. LayerNorm vs. GroupNorm: know *why* BatchNorm breaks with small batch sizes (batch statistics become noisy) and why transformers use LayerNorm instead (no dependency on batch dimension).
- **Optimization** — SGD+momentum vs. Adam/AdamW; learning rate warmup and cosine decay; why weight decay in Adam needed the "decoupled" fix (Loshchilov & Hutter, *AdamW*, ICLR 2019).
- **Transfer learning & fine-tuning** — pretrain on ImageNet/large corpora, fine-tune on your task. Know when to freeze backbone layers vs. fine-tune everything (small dataset → freeze more; large + different domain → fine-tune more).
- **Loss functions for vision** — cross-entropy, focal loss (for class imbalance in detection — Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017), triplet/contrastive loss (for embeddings).

### 3.3 Object detection

Two families, and every interview question maps to "which family, and what's the tradeoff":

- **Two-stage (accuracy-leaning):** R-CNN → Fast R-CNN → **Faster R-CNN** (Ren et al., 2015 — Region Proposal Network replaces selective search, making the whole thing end-to-end trainable). Then Mask R-CNN (He et al., 2017) adds a segmentation branch.
- **One-stage (speed-leaning):** YOLO (Redmon et al., 2016 — detection as single regression problem), SSD (multi-scale feature maps), and the YOLO lineage through YOLOv8/v10/v11 (anchor-free heads, better label assignment).
- **Anchor-free:** FCOS, CenterNet — predict object centers/keypoints directly, removing the anchor-box hyperparameter headache.
- **Transformer-based:** **DETR** (Carion et al., ECCV 2020) — frames detection as set prediction with bipartite matching, removing NMS and anchors entirely. Slow to converge originally; Deformable DETR and DINO-DETR fixed that.

**The question that filters candidates:** "Why does Faster R-CNN need NMS but you might design a detector that doesn't?" Tests whether you understand that NMS is a hand-crafted fix for duplicate predictions, and that set-prediction losses (DETR's Hungarian matching) solve the same problem end-to-end.

### 3.4 Semantic & instance segmentation

FCN (Long et al., 2015 — first fully convolutional, end-to-end segmentation) → U-Net (Ronneberger et al., 2015 — encoder-decoder with skip connections, ubiquitous in medical imaging, which is directly relevant to my own work in biomedical vision) → DeepLab family (atrous/dilated convolution to grow receptive field without losing resolution, plus CRF post-processing in v1/v2) → Mask R-CNN for instance-level masks → **Segment Anything (SAM)**, Kirillov et al., Meta AI, 2023 — a promptable, zero-shot segmentation foundation model trained on 1.1 billion masks, followed by **SAM 2** (2024) extending this to video with real-time tracking.

I use SAM2 directly in my own PhD pipeline — auto-generating segmentation labels for residual-limb tracking, since no public amputee dataset exists. This is a good real illustration of "foundation model as label generator," a pattern you'll see constantly in industry now: use a big general model to bootstrap labels for a small specialized model that actually runs at inference time.

---

## Stage 4 — Modern Era: Transformers, Self-Supervision, Generative & Multimodal Models (Months 6–14)

### 4.1 Vision Transformers

**ViT** (Dosovitskiy et al., *"An Image is Worth 16x16 Words,"* ICLR 2021) splits an image into patches, linearly embeds them, and feeds the sequence to a standard Transformer encoder — no convolutions. The catch: ViT lacks CNN's inductive biases (locality, translation equivariance), so it underperforms CNNs on small data and needs either huge pretraining data (JFT-300M) or heavy augmentation/distillation (**DeiT**, Touvron et al., 2021) to be competitive.

**Swin Transformer** (Liu et al., ICCV 2021, Best Paper) reintroduces a locality bias via shifted windows and hierarchical feature maps, making transformers practical as general-purpose vision backbones (detection, segmentation) rather than just classifiers.

**Interview angle:** "Why does a ViT need more data than a ResNet to reach the same accuracy?" — because convolution *bakes in* translation equivariance and locality as architectural priors; a transformer has to *learn* these from data, which costs samples.

### 4.2 Self-supervised & representation learning

This is arguably the most important shift of the last five years: learning strong visual representations *without* labels.

- **Contrastive methods:** SimCLR (Chen et al., 2020) — pull augmented views of the same image together, push different images apart, needs large batches for enough negatives. MoCo (He et al., 2020) — a momentum-updated memory queue removes the large-batch requirement.
- **Non-contrastive:** BYOL (Grill et al., 2020) — no negatives at all, just a momentum target network and stop-gradient; surprising that this doesn't collapse, and understanding *why* (the predictor + momentum encoder asymmetry) is a great interview discussion.
- **Distillation-based:** DINO / **DINOv2** (Caron et al. / Oquab et al., Meta AI) — self-distillation with no labels produces features so good that frozen DINOv2 features beat many supervised backbones on downstream tasks, and its attention maps emerge as unsupervised object segmentation, unprompted.
- **Masked image modeling:** **MAE** (He et al., CVPR 2022) — mask 75% of image patches, reconstruct pixels; the vision analogue of BERT's masked language modeling, and dramatically more compute-efficient to pretrain than contrastive methods since the encoder only sees the visible patches.

### 4.3 Vision-language and multimodal foundation models

- **CLIP** (Radford et al., OpenAI, 2021) — jointly embeds images and text via contrastive learning on 400M pairs, enabling zero-shot classification with no task-specific fine-tuning at all. This paper reframed "what is a vision model for" across the whole field.
- **BLIP / BLIP-2** — bridges frozen vision encoders and frozen LLMs with a lightweight querying transformer, a template later reused across the field.
- **LLaVA and GPT-4V/GPT-5-class, Gemini, Claude multimodal models** — connect a vision encoder to a large language model for open-ended visual reasoning, captioning, and instruction-following over images.
- Know the practical distinction interviewers probe: **CLIP-style contrastive models** are for retrieval/zero-shot classification; **LLaVA-style models** are for open-ended reasoning and generation about image content. Different objective, different use case — don't conflate them.

### 4.4 Generative vision models

- **GANs** (Goodfellow et al., 2014) — generator vs. discriminator in a minimax game; StyleGAN (Karras et al.) refined this to photorealistic, controllable face synthesis via a mapping network and adaptive instance normalization. GANs are notoriously unstable to train (mode collapse) — know *why*: the discriminator can win too early and provide no useful gradient.
- **VAEs** — an encoder maps to a latent distribution, a decoder reconstructs; trained with a reconstruction loss plus a KL term regularizing the latent space toward a prior. Blurrier outputs than GANs, but a stable, well-behaved latent space (useful for interpolation).
- **Diffusion models** — the current generative state of the art. DDPM (Ho et al., 2020) formalizes generation as learning to reverse a gradual noising process; the model at each step predicts the noise to remove. **Stable Diffusion** (Rombach et al., CVPR 2022) made this tractable by running the diffusion process in a compressed latent space rather than pixel space. Current frontier: video diffusion models (Sora-class systems) extending this to temporally consistent video generation.

**Why diffusion beat GANs for most applications:** stable training (a straightforward denoising regression loss, no adversarial game to balance), better mode coverage, and it composes naturally with classifier-free guidance for controllable, text-conditioned generation.

---

## Stage 5 — Specialization & Systems Thinking (Ongoing)

Pick a specialization once you're through Stage 4 — depth over breadth is what gets you into a research role or a senior IC role. Common tracks:

- **3D vision / NeRF & Gaussian Splatting** — Neural Radiance Fields (Mildenhall et al., 2020) represent a scene as a continuous volumetric function queried by a small MLP; 3D Gaussian Splatting (Kerbl et al., 2023) trades that for explicit, rasterizable Gaussians, achieving real-time rendering — the current preferred approach for novel-view synthesis in production.
- **Video understanding** — 3D CNNs (C3D, I3D) → two-stream networks → video transformers (TimeSformer, VideoMAE) for action recognition and temporal reasoning.
- **Efficient / edge deployment** — quantization (INT8/INT4), structured pruning, knowledge distillation, MobileNet/ShuffleNet-style depthwise-separable convolutions. This is exactly the world I live in for my own work: my segmentation model has to run in real time (<90ms end-to-end) on embedded hardware for prosthetic control, so distillation and quantization aren't academic exercises — they're the difference between a system that works on a patient's arm and one that doesn't.
- **Medical/biomedical imaging** — domain shift, small-data regimes, U-Net variants, and the specific ethics/validation bar of clinical deployment (IRB approval, sensitivity/specificity trade-offs matter more than top-1 accuracy).
- **Robotics/SLAM** — visual-inertial odometry, loop closure, combining classical geometry (Stage 2) with learned feature matching (SuperPoint/SuperGlue).

---

## Current State of the Art (2025–2026 snapshot)

- **Segmentation:** SAM2 and its derivatives dominate promptable segmentation, including video, in near real-time.
- **Representation learning:** DINOv2-class self-supervised backbones are now standard "foundation" vision encoders, often outperforming supervised pretraining for downstream transfer.
- **Multimodal reasoning:** frontier multimodal LLMs (GPT, Gemini, Claude vision-language lines) handle document understanding, chart/diagram reasoning, and grounded visual QA approaching specialist-model performance, largely closing the gap for many previously bespoke tasks.
- **Generation:** diffusion-based image and video models continue to improve in temporal consistency and controllability; latent-space diffusion remains the dominant recipe for compute efficiency.
- **Efficient deployment:** the field has shifted from "train a huge model" to "train or distill a huge model, then compress it" as the default two-stage recipe for anything shipping to a device.
- **3D:** Gaussian Splatting has largely displaced vanilla NeRF for real-time applications, with active research on dynamic (4D) scenes.

The throughline across all of these: **general-purpose foundation models pretrained at scale, then adapted cheaply** (fine-tuning, prompting, distillation, LoRA-style adapters) has replaced "train a bespoke model per task" as the default paradigm. If you understand *why* that shift happened — pretraining amortizes the expensive part of learning general visual structure across every downstream task — you understand where the field is heading, not just where it's been.

---

## How Companies Actually Interview for This

After sitting on both sides of this table, the questions cluster into four buckets:

1. **Fundamentals, derived not recalled** — "Why is convolution equivariant but pooling isn't?" "Derive the receptive field of a 3-layer 3×3 conv stack." "Why does batch norm fail with batch size 1?" They're checking you understand mechanism, not vocabulary.
2. **Coding, usually classical CV or basic tensor ops** — implement non-max suppression, IoU, a sliding-window convolution, or connected-component labeling from scratch in plain NumPy/Python — no framework crutch.
3. **System design for vision** — "Design a real-time face recognition system for a building entrance" or "Design a defect-detection pipeline for a manufacturing line." They want you to reason about the full pipeline: data collection and labeling cost, model choice given latency/accuracy/hardware constraints, handling class imbalance and distribution shift, and monitoring after deployment — not just "use ResNet."
4. **Paper discussion** — pick 2–3 papers you can discuss at a level deeper than the abstract: what problem existed before it, what specifically it changed, what its limitation was, and what paper fixed *that*. I'd pick one from each stage above (e.g., ResNet, one self-supervised method, one from your specialization) rather than trying to cover everything shallowly.

**My honest advice after years of both taking and giving these interviews:** breadth gets you through the resume screen; depth on a handful of topics you can defend under follow-up questions is what gets you the offer. Pick three things from this roadmap and go deep enough that I couldn't rattle you with a follow-up question. That's a better use of your last two weeks before an interview than skimming everything one more time.

---

*Have a specific topic you want unpacked further, or a paper you'd like a plain-language walkthrough of? [Reach out](/contact/) — I enjoy these conversations more than almost anything else about this field.*
