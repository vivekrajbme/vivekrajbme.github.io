---
layout: single
title: "CV Stage 1: Foundations"
permalink: /learn-computer-vision/stage-1-foundations/
author_profile: true
author: vivek-raj
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: "Image formation, sampling, linear algebra, and classical image processing — built from scratch."
---

[← Back to the full curriculum](/learn-computer-vision/)

Everything here should be built once with plain NumPy, no OpenCV. That's not a purity test — it's the only way to be sure you understand the *mechanism* rather than the API call.

---

## 1.1 How an Image Is Formed

### The pinhole camera model

A 3D point $(X, Y, Z)$ projects onto the image plane at:

$$x = f\frac{X}{Z}, \qquad y = f\frac{Y}{Z}$$

In homogeneous coordinates, the full pipeline from world point to pixel is:

$$\begin{pmatrix}x\\y\\1\end{pmatrix} \sim K\,[R\,|\,t]\,\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}$$

- $[R|t]$ (**extrinsics**) moves a point from world coordinates into the camera's own coordinate frame.
- $K$ (**intrinsics**) — focal length and principal point — projects that camera-frame point onto the pixel grid.

Every point along the ray through $(X,Y,Z)$ maps to the same pixel — depth is destroyed by a single view. That's *the* reason monocular depth is ambiguous and why stereo (Stage 2) and multi-view geometry exist.

```python
import numpy as np

def project_points(points_3d, K, R, t):
    """points_3d: (N,3) world points. Returns (N,2) pixel coords."""
    cam_pts = (R @ points_3d.T + t.reshape(3, 1)).T   # world -> camera frame
    proj = (K @ cam_pts.T).T                          # camera -> homogeneous pixels
    return proj[:, :2] / proj[:, 2:3]                 # de-homogenize

K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0,   0,   1]], dtype=float)
R = np.eye(3)
t = np.array([0, 0, 5.0])                             # camera is 5 units back
pts_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
print(project_points(pts_3d, K, R, t))
```

### Sampling and aliasing

By the Nyquist-Shannon theorem, a signal sampled below twice its highest frequency cannot be reconstructed — high frequencies fold back and appear as false low frequencies. That's why photographing a monitor or fine fabric produces moiré patterns, and why real cameras put an optical low-pass (anti-aliasing) filter in front of the sensor.

This isn't just optics trivia: Zhang, *"Making Convolutional Networks Shift-Invariant Again"* (ICML 2019), showed strided downsampling in CNNs aliases feature maps the same way, which is why max-pooling doesn't give you the shift-invariance people assume it does.

```python
# Demonstrate aliasing: sample a high-frequency sine below Nyquist
import numpy as np
t = np.linspace(0, 1, 1000)
true_signal = np.sin(2 * np.pi * 40 * t)     # 40 Hz signal
fs = 50                                       # sample at 50 Hz -> Nyquist = 25 Hz, too low
sample_idx = np.arange(0, len(t), int(1000 / fs))
aliased = true_signal[sample_idx]
# Reconstructing from these samples will show a much LOWER apparent frequency (10 Hz)
```

### Quantization

Continuous intensity is rounded to discrete levels — 8-bit gives 256 levels/channel. Too coarse on a smooth gradient → visible **banding**, the reason HDR/16-bit pipelines exist for high-dynamic-range scenes.

---

## 1.2 Linear Algebra & Probability, with a CV Lens

### Eigen-decomposition / SVD → PCA (Eigenfaces)

Any covariance matrix decomposes into orthogonal directions of maximum variance. Classic worked example: represent any face as a weighted sum of a handful of "eigenfaces."

```python
import numpy as np

def eigenfaces(face_vectors, k=50):
    """face_vectors: (N, D) flattened face images. Returns top-k eigenfaces + mean."""
    mean_face = face_vectors.mean(axis=0)
    centered = face_vectors - mean_face
    # SVD is more numerically stable than eigen-decomposing the covariance directly
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenfaces = Vt[:k]                      # top-k principal directions
    weights = centered @ eigenfaces.T        # project faces onto eigenface basis
    return mean_face, eigenfaces, weights

def reconstruct(mean_face, eigenfaces, weights):
    return mean_face + weights @ eigenfaces
```

The same idea — keep the low-rank directions that explain the most variance — is exactly what LoRA does when fine-tuning a huge model's weight matrix today: it doesn't touch the full matrix, only a low-rank correction to it. One idea, one century apart in application.

### Convolution as matrix multiplication (im2col)

```python
import numpy as np

def im2col(image, kh, kw):
    """Unroll every kh x kw patch into a row -> convolution becomes one matmul."""
    H, W = image.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    cols = np.zeros((out_h * out_w, kh * kw))
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            cols[idx] = image[i:i+kh, j:j+kw].flatten()
            idx += 1
    return cols, (out_h, out_w)

def conv2d_via_matmul(image, kernel):
    kh, kw = kernel.shape
    cols, (out_h, out_w) = im2col(image, kh, kw)
    result = cols @ kernel.flatten()
    return result.reshape(out_h, out_w)
```

This is literally what GPUs execute for every conv layer — and why depthwise/grouped convolutions (MobileNet) are fast: smaller matrices, less redundant unrolling.

### Bayes' rule / MAP estimation

$$P(\text{signal} \mid \text{observation}) \propto P(\text{observation} \mid \text{signal}) \cdot P(\text{signal})$$

This single line underlies image denoising, Kalman filtering (Stage 2), and diffusion models (Stage 4) — the last of which learns $\nabla_x \log p(x)$, the *score*, precisely because Bayes' rule makes that gradient sufficient to sample from the distribution.

---

## 1.3 Classical Image Processing, Built by Hand

### Convolution / correlation

```python
import numpy as np

def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out
```

### Separable Gaussian blur

A 2D Gaussian factors as $G(x,y) = G(x) \cdot G(y)$, turning an $O(n^2)$ blur into two $O(n)$ passes — a real algorithmic speedup, not a micro-optimization.

```python
def gaussian_kernel_1d(size, sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def separable_gaussian_blur(image, size=5, sigma=1.0):
    k = gaussian_kernel_1d(size, sigma)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, k, mode='same'), axis=1, arr=image)
    blurred = np.apply_along_axis(lambda col: np.convolve(col, k, mode='same'), axis=0, arr=blurred)
    return blurred
```

### Canny edge detection, from scratch — with the *why* at each stage

```python
import numpy as np

def sobel_gradients(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = Kx.T
    gx = convolve2d(image, Kx)
    gy = convolve2d(image, Ky)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    """Thin the blurry gradient ridge to single-pixel edges."""
    H, W = magnitude.shape
    out = np.zeros_like(magnitude)
    angle = np.rad2deg(direction) % 180
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            # pick the two neighbors along the gradient direction
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                out[i, j] = magnitude[i, j]
    return out

def hysteresis_threshold(image, low, high):
    """Two thresholds fix edge fragmentation that one threshold can't."""
    strong = image >= high
    weak = (image >= low) & (image < high)
    result = np.zeros_like(image)
    result[strong] = 1
    # keep a weak edge only if it touches a strong edge
    H, W = image.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak[i, j] and strong[i-1:i+2, j-1:j+2].any():
                result[i, j] = 1
    return result

def canny(image, sigma=1.0, low=0.1, high=0.3):
    smoothed = separable_gaussian_blur(image, size=5, sigma=sigma)   # stage 1: suppress noise
    magnitude, direction = sobel_gradients(smoothed)                 # stage 2: find gradients
    thinned = non_max_suppression(magnitude, direction)              # stage 3: thin to 1px
    return hysteresis_threshold(thinned / thinned.max(), low, high)  # stage 4: fix fragmentation
```

**Why each stage exists, in one line each:** smoothing trades a little localization for a lot of noise robustness; the gradient alone gives you a blurry ridge, not an edge; non-max suppression thins that ridge to one pixel by keeping only the local maximum along the gradient direction; a single threshold either lets noise through or breaks real edges into dashed fragments, so hysteresis keeps weak pixels *only* when they connect to a strong one.

### Histogram equalization

```python
def histogram_equalize(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return equalized.reshape(image.shape).astype(np.uint8)
```

### Morphological operations

```python
from scipy.ndimage import binary_erosion, binary_dilation

def opening(mask, structure=None):   # erode then dilate: removes small noise specks
    return binary_dilation(binary_erosion(mask, structure), structure)

def closing(mask, structure=None):   # dilate then erode: fills small holes
    return binary_erosion(binary_dilation(mask, structure), structure)
```

You'll reach for exactly this vocabulary when cleaning up a neural network's segmentation mask — the classical toolkit didn't disappear, it moved downstream to post-processing.

### Fourier transform of an image

```python
import numpy as np

def fourier_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)          # move zero-frequency to the center
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum
```

Low frequencies = overall shape/smooth regions; high frequencies = edges and texture. This is why JPEG can discard high-frequency coefficients with little perceived loss — and it's the basis for Geirhos et al.'s well-known finding that ImageNet-trained CNNs are texture-biased rather than shape-biased, unlike humans.

---

## Exercises

1. Implement `canny()` above on a real grayscale image and tune `low`/`high` until edges look clean — notice how much the result depends on those two numbers.
2. Prove to yourself that separable convolution gives the *identical* numerical result as full 2D convolution, just faster.
3. Take an image, zero out its high-frequency Fourier coefficients, and inverse-transform it. What do you see, and why does this match what a Gaussian blur does in the spatial domain?

---

## Resources

**Courses**
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/) — the standard starting course; the [lecture notes](https://cs231n.github.io/) alone are worth reading end to end.
- *"First Principles of Computer Vision"* (Shree Nayar, Columbia) — search this title on YouTube for the best available video treatment of image formation, sampling, and camera optics from physical first principles.

**Papers**
- Zhang, *"Making Convolutional Networks Shift-Invariant Again"*, ICML 2019 — [arXiv:1904.11486](https://arxiv.org/abs/1904.11486)
- Geirhos et al., *"ImageNet-trained CNNs are biased towards texture"*, ICLR 2019 — [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)

**Sites / tools**
- [PyImageSearch](https://pyimagesearch.com/) — practical, code-first classical CV tutorials.
- [OpenCV documentation](https://docs.opencv.org/) — once you've built it by hand, this is where you go for the production version.

---

[Next: Stage 2 — Classical Computer Vision →](/learn-computer-vision/stage-2-classical-cv/)
