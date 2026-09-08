---
layout: single
title: "CV Stage 3: Deep Learning for Vision"
permalink: /learn-computer-vision/stage-3-deep-learning/
author_profile: true
author: vivek-raj
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: "CNNs, training mechanics, object detection, and segmentation."
---

[← Back to the full curriculum](/learn-computer-vision/) · [← Stage 2: Classical CV](/learn-computer-vision/stage-2-classical-cv/)

This is where most self-taught engineers start, and where most of them develop gaps because they skipped Stages 1–2. If you did those stages, this section will feel like it clicks into place rather than arriving from nowhere.

---

## 3.1 The Convolutional Neural Network, Properly Understood

A conv layer does three things at once:

1. **Local connectivity** — each output depends on a small receptive field, not the whole image.
2. **Weight sharing** — the same filter slides everywhere, giving translation *equivariance* (shift the input, the output shifts identically).
3. **Hierarchical composition** — stacking layers grows the receptive field and the abstraction level, from edges to textures to parts to objects.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
```

**The receptive field question, worked out:** stack three 3×3 convolutions and the effective receptive field is $3 + 2(3-1) = 7$ — the same coverage as one 7×7 conv, but with three ReLUs (more non-linearity) and far fewer parameters ($3 \times 3^2 = 27$ vs. $7^2 = 49$ per channel pair). This is the exact argument VGG made in 2014 and it's still a standard interview derivation.

### Architecture lineage — know what each one *introduced*

| Year | Model | What it introduced |
|---|---|---|
| 1998 | LeNet-5 | Conv + pooling for digit recognition — the template |
| 2012 | AlexNet | ReLU, dropout, GPU training — ignited the deep learning era |
| 2014 | VGG | Stacked 3×3 convs: bigger receptive field, fewer parameters |
| 2014 | GoogLeNet/Inception | Multi-scale filters in parallel; 1×1 convs for cheap dimensionality reduction |
| 2015 | **ResNet** | Residual/skip connections — arguably the single most important architectural idea in deep learning |
| 2017 | DenseNet | Feature reuse via dense connections |
| 2019 | EfficientNet | Compound scaling of depth/width/resolution via neural architecture search |
| 2020 | Vision Transformer | Images as patch sequences — no convolutional inductive bias at all |

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    """The idea that made 100+ layer networks trainable."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)   # the skip connection
```

**The ResNet question you must be able to answer:** why do skip connections help? Because the identity path lets gradients flow directly through backprop without vanishing, *and* because the network now only has to learn a residual $F(x) = H(x) - x$ rather than the full mapping $H(x)$ — an easier optimization target. This is asked in nearly every CV interview.

---

## 3.2 Training Mechanics

```python
import torch
import torchvision.transforms as T

# Augmentation: encoding invariances you WANT the model to have, not a checklist
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def mixup(x, y, alpha=0.2):
    """Blend two images and their labels; regularizes decision boundaries."""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
```

- **Normalization:** BatchNorm normalizes across the *batch* dimension — this breaks down at small batch sizes because batch statistics become noisy estimates. LayerNorm normalizes across *features* instead, with no dependency on batch size, which is exactly why Transformers (Stage 4) use LayerNorm rather than BatchNorm.
- **Optimization:** SGD+momentum generalizes slightly better on large-scale image classification; Adam/AdamW converges faster and is the default almost everywhere else. AdamW (Loshchilov & Hutter, ICLR 2019) fixed a real bug in Adam — weight decay was implicitly scaled by the adaptive learning rate, which AdamW decouples.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

- **Transfer learning:** small target dataset → freeze most of the backbone and fine-tune only the head; large target dataset from a different domain → fine-tune the whole network at a lower learning rate.

```python
import torchvision.models as models

model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False              # freeze backbone
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace + train only this head
```

- **Focal loss** (Lin et al., ICCV 2017) down-weights easy, well-classified examples so a detector's loss isn't dominated by the overwhelming number of easy background boxes:

```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()
```

---

## 3.3 Object Detection

Two families — every question maps to "which family, what's the tradeoff":

- **Two-stage (accuracy-leaning):** R-CNN → Fast R-CNN → **Faster R-CNN** (a Region Proposal Network replaces selective search, making the whole pipeline end-to-end trainable) → Mask R-CNN adds a segmentation branch on top.
- **One-stage (speed-leaning):** YOLO (detection as one regression problem), SSD (multi-scale feature maps), through the YOLOv8/v10/v11 lineage (anchor-free heads, better label assignment).
- **Anchor-free:** FCOS, CenterNet — predict object centers directly, removing the anchor-box hyperparameter headache.
- **Transformer-based:** **DETR** frames detection as set prediction with bipartite (Hungarian) matching, removing NMS and anchors entirely — slow to converge originally, fixed by Deformable DETR.

```python
import torchvision

# Two-stage detector, pretrained, ready for inference/fine-tuning
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()
predictions = model([image_tensor])  # list of dicts: boxes, labels, scores

# Non-max suppression from scratch — the classic detection interview question
def nms(boxes, scores, iou_threshold=0.5):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep

def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)
```

**The question that filters candidates:** "Why does Faster R-CNN need NMS but you might design a detector that doesn't?" NMS is a hand-crafted fix for duplicate predictions; DETR's set-prediction loss solves the same problem end-to-end via bipartite matching during training.

---

## 3.4 Semantic and Instance Segmentation

FCN (first fully-convolutional, end-to-end segmentation) → **U-Net** (encoder-decoder with skip connections, ubiquitous in medical imaging) → DeepLab family (atrous/dilated convolution grows receptive field without losing resolution) → Mask R-CNN (instance-level masks) → **Segment Anything (SAM / SAM2)** — a promptable, zero-shot segmentation foundation model.

```python
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNet(nn.Module):
    """The essential skip-connection idea, without the full 4-level depth."""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)   # 128 = 64 (upsampled) + 64 (skip connection)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))   # the skip connection: recover fine detail
        return self.out(d1)
```

```python
# Using Segment Anything (SAM) for zero-shot, promptable segmentation
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(point_coords=[[x, y]], point_labels=[1])
```

I use SAM2 in exactly this way in my own PhD pipeline: auto-generating segmentation labels for residual-limb tracking, since no public amputee dataset exists — a good concrete illustration of "use a big general model to bootstrap labels for a small specialized model that runs at inference time." That pattern (big model labels, small model deploys) is now standard practice across the industry, not a research curiosity.

---

## Exercises

1. Train `SimpleCNN` above on CIFAR-10 and plot train/val accuracy with and without data augmentation — quantify the gap.
2. Implement NMS from scratch (above) and verify it against `torchvision.ops.nms` on the same boxes and scores.
3. Fine-tune a pretrained `fasterrcnn_resnet50_fpn` on a small custom dataset (even 50–100 labeled images) using `torchvision`'s detection reference training script, and explain what changes when you freeze vs. unfreeze the backbone.

---

## Resources

**Papers (all well worth reading in full, in this order)**
- Krizhevsky et al., *AlexNet*, NeurIPS 2012.
- Simonyan & Zisserman, *VGG* — [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He et al., *Deep Residual Learning (ResNet)* — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Ren et al., *Faster R-CNN* — [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Redmon et al., *YOLO* — [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- Lin et al., *Focal Loss / RetinaNet* — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Carion et al., *DETR* — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Long et al., *Fully Convolutional Networks* — [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
- Ronneberger et al., *U-Net* — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Kirillov et al., *Segment Anything* — [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- Loshchilov & Hutter, *Decoupled Weight Decay (AdamW)* — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

**Courses**
- [Stanford CS231n](https://cs231n.stanford.edu/) — still the best structured path through this stage.
- [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) — code-first, gets you training real models in week one.
- Andrej Karpathy's YouTube channel, [@AndrejKarpathy](https://www.youtube.com/@AndrejKarpathy) — his "from scratch" build videos are the closest thing to a live mentoring session on this material.

**Sites / code**
- [Papers with Code](https://paperswithcode.com/) — leaderboards, and almost every architecture above has a reference implementation linked from there.
- [torchvision model zoo](https://pytorch.org/vision/stable/models.html) — pretrained ResNet/Faster R-CNN/Mask R-CNN ready to fine-tune.

---

[Next: Stage 4 — Transformers, Self-Supervision, and Generative Models →](/learn-computer-vision/stage-4-modern-era/)
