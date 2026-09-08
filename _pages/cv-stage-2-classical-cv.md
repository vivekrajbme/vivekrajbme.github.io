---
layout: single
title: "CV Stage 2: Classical Computer Vision"
permalink: /learn-computer-vision/stage-2-classical-cv/
author_profile: true
author: vivek-raj
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: "Feature detection, geometric vision, and pre-deep-learning segmentation."
---

[← Back to the full curriculum](/learn-computer-vision/) · [← Stage 1: Foundations](/learn-computer-vision/stage-1-foundations/)

This is the stage that still gets asked about in interviews for AR/VR, robotics, and camera-pipeline roles — deep learning didn't erase this knowledge, it moved it into "explain what a CNN is implicitly learning."

---

## 2.1 Feature Detection and Description

### Harris corners

A corner is a point where the local autocorrelation changes in *all* directions. Formally, for a small shift $(u,v)$, the sum of squared differences is approximated by:

$$E(u,v) \approx \begin{pmatrix}u & v\end{pmatrix} M \begin{pmatrix}u\\v\end{pmatrix}, \qquad M = \sum_{x,y} w(x,y)\begin{pmatrix}I_x^2 & I_xI_y \\ I_xI_y & I_y^2\end{pmatrix}$$

If both eigenvalues of $M$ are large, you have a corner (intensity changes fast in every direction); one large and one small means an edge; both small means a flat region.

```python
import numpy as np

def harris_response(image, k=0.04, window=5):
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)
    Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix * Iy
    from scipy.ndimage import uniform_filter
    Sxx = uniform_filter(Ixx, window)
    Syy = uniform_filter(Iyy, window)
    Sxy = uniform_filter(Ixy, window)
    det = Sxx * Syy - Sxy**2
    trace = Sxx + Syy
    return det - k * trace**2     # Harris response R; corners have R > threshold
```

### SIFT, HOG, ORB — what each one actually solves

- **SIFT** (Lowe, 2004) finds extrema across a *scale-space* built from Difference-of-Gaussians, then assigns each keypoint a dominant orientation from local gradients before building its descriptor — that's where scale- and rotation-invariance come from, not from magic. One paper, thirty-thousand-plus citations, and it single-handedly powered a decade of panorama stitching and image retrieval.
- **HOG** (Dalal & Triggs, CVPR 2005) divides the image into cells, builds a histogram of gradient orientations per cell, and normalizes over overlapping blocks — the descriptor behind the first real-time pedestrian detectors.
- **ORB / FAST / BRIEF** trade descriptor quality for raw speed via binary descriptors, which is why they're the default in visual-inertial odometry on drones and phones — cheap enough to run at 200 Hz on embedded hardware.

```python
import cv2

img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(img, None)

# Matching two images' descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_orb, des_orb)  # replace second arg with a second image's descriptors
matches = sorted(matches, key=lambda m: m.distance)
```

**Interview framing:** "Explain SIFT's scale invariance" is testing whether you understand *why* Gaussian scale-space plus extrema detection gives invariance — not whether you memorized the pipeline diagram.

---

## 2.2 Geometric Vision

### Camera calibration

Zhang's checkerboard method (2000) is still the industry standard: show the camera a flat checkerboard from several angles, and solve for the intrinsic matrix $K$ and distortion coefficients that make all the observed corners consistent.

```python
import cv2
import numpy as np

# Standard OpenCV calibration flow
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)   # 9x6 checkerboard, one square = one unit

objpoints, imgpoints = [], []   # collect these across multiple checkerboard images
for img_gray in checkerboard_images:                # list of grayscale images
    found, corners = cv2.findChessboardCorners(img_gray, (9, 6))
    if found:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_gray.shape[::-1], None, None
)
```

### Epipolar geometry

The **essential matrix** $E$ (calibrated cameras) or **fundamental matrix** $F$ (uncalibrated) encodes the constraint that a point in one image must lie on a specific line — the *epipolar line* — in the other image. This single constraint is what makes stereo matching tractable: instead of searching the whole second image for a correspondence, you search one line.

```python
import cv2
import numpy as np

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)  # pts1/pts2: matched keypoints
# Epipolar line in image 2 for a point in image 1:
line2 = F @ np.array([pts1[0][0], pts1[0][1], 1.0])
```

### Stereo vision and disparity

Depth is inversely proportional to disparity: $Z = \dfrac{fB}{d}$, where $B$ is the baseline (distance between the two cameras) and $d$ is the pixel disparity of the same point between the two views.

```python
import cv2

stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
depth = (focal_length * baseline) / (disparity + 1e-6)
```

Classical block-matching stereo (SGM) is still shipped in real products today, because it's deterministic and doesn't need a GPU — a good reminder that "solved by deep learning" doesn't mean "the classical method disappeared."

### Optical flow

**Lucas-Kanade** (local, sparse) assumes flow is locally constant in a small window and solves a small least-squares system per point; **Horn-Schunck** (global, dense) instead regularizes flow smoothness over the whole image. Both set up the intuition for modern learned flow networks (FlowNet, RAFT) in Stage 4/5.

```python
import cv2
import numpy as np

p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.3, minDistance=7)
p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)
good_new, good_old = p1[status == 1], p0[status == 1]   # flow vectors: good_new - good_old
```

---

## 2.3 Segmentation and Clustering (Pre-Deep-Learning)

Know these exist and their failure modes — many "why did DeepLab do X" questions are really "what problem did classical segmentation have that this solves."

```python
import cv2
import numpy as np
from skimage.segmentation import slic, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# GrabCut: interactive foreground extraction via a user-provided rectangle
mask = np.zeros(img.shape[:2], np.uint8)
bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
rect = (50, 50, 300, 400)
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# SLIC superpixels: over-segment into perceptually meaningful groups
segments = slic(img, n_segments=200, compactness=10)

# Watershed: treats the gradient image as a topographic surface and floods it from markers
distance = ndi.distance_transform_edt(binary_mask)
local_max = peak_local_max(distance, min_distance=20)
markers = ndi.label(local_max)[0]
labels = watershed(-distance, markers, mask=binary_mask)
```

**Their common failure mode:** all of these rely on hand-crafted cues (color, gradient, distance) with no learned semantics — they can't tell "dog" from "sofa," only "this region looks locally different from that region." That gap is exactly what FCN/U-Net/Mask R-CNN (Stage 3) were built to close.

---

## Exercises

1. Implement Harris corner detection from scratch and compare its output to `cv2.cornerHarris` on the same image.
2. Calibrate a real camera (print a checkerboard, take 15–20 photos) and undistort an image using the resulting `K` and distortion coefficients.
3. Compute stereo disparity on a rectified stereo pair and convert it to a metric depth map — then explain why disparity near the camera is more accurate than disparity far away (hint: look at the $Z = fB/d$ relationship's sensitivity to small errors in $d$).

---

## Resources

**Papers**
- Lowe, *"Distinctive Image Features from Scale-Invariant Keypoints"*, IJCV 2004 — [David Lowe's SIFT page](https://www.cs.ubc.ca/~lowe/keypoints/) (original paper + code).
- Dalal & Triggs, *"Histograms of Oriented Gradients for Human Detection"*, CVPR 2005.
- Zhang, *"A Flexible New Technique for Camera Calibration"*, IEEE TPAMI 2000 — the checkerboard calibration method every library implements.

**Books**
- Richard Szeliski, *Computer Vision: Algorithms and Applications* — free to read online at [szeliski.org/Book](https://szeliski.org/Book/), the single best reference for this entire stage.
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* — the definitive (and famously dense) reference on epipolar geometry and multi-view geometry.

**Courses / sites**
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [PyImageSearch](https://pyimagesearch.com/) for hands-on classical CV code walkthroughs.

---

[Next: Stage 3 — Deep Learning for Vision →](/learn-computer-vision/stage-3-deep-learning/)
