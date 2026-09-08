---
layout: single
title: "CV Stage 4: Transformers, Self-Supervision & Generative Models"
permalink: /learn-computer-vision/stage-4-modern-era/
author_profile: true
author: vivek-raj
toc: true
toc_label: "On this page"
toc_sticky: true
excerpt: "Vision Transformers, self-supervised learning, vision-language models, and generative vision."
---

[← Back to the full curriculum](/learn-computer-vision/) · [← Stage 3: Deep Learning](/learn-computer-vision/stage-3-deep-learning/)

This is the biggest paradigm shift of the last five years: from "train a bespoke model per task" to "pretrain a general-purpose model at scale, adapt it cheaply." Everything in this stage is an instance of that shift.

---

## 4.1 Vision Transformers

**ViT** (Dosovitskiy et al., ICLR 2021) splits an image into fixed-size patches, linearly embeds each as a token, and feeds the sequence to a standard Transformer encoder — no convolutions anywhere.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        # a strided conv here is just an efficient way to do "slice into patches + linear embed"

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, H/patch, W/patch)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

class MinimalViT(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))  # 196 patches + CLS
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])   # classify from the CLS token
```

**Why ViT needs more data than a ResNet to match its accuracy:** convolution *bakes in* translation equivariance and locality as architectural priors; a Transformer has no such prior and has to *learn* it from data — that costs samples. This is the standard interview follow-up to "what's the difference between ViT and ResNet."

**Swin Transformer** reintroduces a locality bias via shifted windows and hierarchical feature maps, which is why it (not vanilla ViT) became the default transformer backbone for detection and segmentation.

---

## 4.2 Self-Supervised & Representation Learning

Learning strong visual representations *without* labels — arguably the most consequential shift of the last five years.

### Contrastive: SimCLR

Pull two augmented views of the same image together in embedding space, push different images apart.

```python
import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """SimCLR's normalized temperature-scaled cross-entropy loss."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    N = z1.size(0)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)])  # each sample's positive pair
    mask = torch.eye(2 * N, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))   # exclude self-similarity
    return F.cross_entropy(sim, labels)
```

SimCLR needs large batches to have enough negatives; **MoCo** fixes this with a momentum-updated memory queue of negatives instead of relying on batch size.

### Non-contrastive: BYOL / DINO

**BYOL** uses no negatives at all — just an online network, a momentum-averaged target network, and a stop-gradient on the target. The surprising part is that this *doesn't* collapse to a trivial constant output; the asymmetry between the predictor (only on the online branch) and the momentum update is what prevents collapse — a genuinely good interview discussion topic.

**DINO / DINOv2** use self-distillation with no labels at all, and the resulting frozen features are strong enough that DINOv2 beats many supervised backbones on downstream transfer — its attention maps emerge as unsupervised object segmentation, entirely unprompted.

### Masked image modeling: MAE

The vision analogue of BERT's masked language modeling: mask ~75% of image patches, reconstruct the missing pixels.

```python
def random_masking(patches, mask_ratio=0.75):
    """patches: (B, N, D). Returns visible patches + info to restore full sequence order."""
    B, N, D = patches.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    visible = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return visible, ids_shuffle, len_keep
```

Because the encoder only ever processes the *visible* patches (25% of them), MAE pretraining is dramatically more compute-efficient than contrastive methods that need to encode full images through two augmented views.

---

## 4.3 Vision-Language and Multimodal Foundation Models

**CLIP** (Radford et al., 2021) jointly embeds images and text with a contrastive loss over 400M pairs, enabling zero-shot classification with no task-specific fine-tuning.

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")
image = preprocess(pil_image).unsqueeze(0)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits = (image_features @ text_features.T).softmax(dim=-1)  # zero-shot classification
```

**BLIP / BLIP-2** bridge a frozen vision encoder and a frozen LLM with a lightweight querying transformer — a template later reused across the field. **LLaVA**-style models connect a vision encoder directly to an LLM's input space for open-ended visual reasoning and instruction-following.

**The practical distinction interviewers probe:** CLIP-style contrastive models are for retrieval and zero-shot classification; LLaVA-style models are for open-ended reasoning and generation about image content. Different training objective, different deployment use case — don't conflate the two.

---

## 4.4 Generative Vision Models

### GANs

Generator vs. discriminator in a minimax game. Notoriously unstable to train — the discriminator can win too early and stop providing a useful gradient (mode collapse).

```python
# One GAN training step, the essential adversarial loop
d_loss = -torch.mean(torch.log(D(real)) + torch.log(1 - D(G(noise).detach())))
d_optimizer.zero_grad(); d_loss.backward(); d_optimizer.step()

g_loss = -torch.mean(torch.log(D(G(noise))))   # generator wants D(fake) -> 1
g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()
```

**StyleGAN** refines this into photorealistic, controllable face synthesis via a mapping network and adaptive instance normalization for style control at different resolutions.

### VAEs

An encoder maps to a latent *distribution*; a decoder reconstructs from a sample of it; trained with reconstruction loss plus a KL term pulling the latent toward a prior.

```python
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
```

Blurrier outputs than GANs, but a well-behaved, interpolatable latent space.

### Diffusion models

The current generative state of the art. DDPM formalizes generation as learning to reverse a gradual noising process — at each step, predict the noise to remove.

```python
def diffusion_training_step(model, x0, timesteps, noise_schedule):
    t = torch.randint(0, timesteps, (x0.size(0),))
    noise = torch.randn_like(x0)
    alpha_bar_t = noise_schedule[t].view(-1, 1, 1, 1)
    x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise  # forward: add noise
    predicted_noise = model(x_t, t)                                   # reverse: predict it
    return F.mse_loss(predicted_noise, noise)
```

**Stable Diffusion** made this tractable by running the diffusion process in a compressed *latent* space (via a pretrained autoencoder) rather than raw pixel space — the same generation quality at a fraction of the compute.

**Why diffusion displaced GANs for most applications:** stable training (a plain denoising regression loss, no adversarial game to balance), better mode coverage, and it composes naturally with classifier-free guidance for controllable, text-conditioned generation.

---

## Exercises

1. Train a small ViT and a small ResNet with the same parameter budget on a small dataset (e.g., CIFAR-100) — observe how much more the ViT benefits from heavier augmentation.
2. Implement the SimCLR loss above and verify pretraining features transfer better than random init on a small downstream classification task.
3. Load CLIP and build a tiny zero-shot classifier for a category it was never explicitly fine-tuned on — try both good and deliberately misleading text prompts and see how sensitive the result is to prompt wording.

---

## Resources

**Papers**
- Dosovitskiy et al., *ViT* — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Liu et al., *Swin Transformer* — [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Chen et al., *SimCLR* — [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Grill et al., *BYOL* — [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Oquab et al., *DINOv2* — [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- He et al., *Masked Autoencoders (MAE)* — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Radford et al., *CLIP* — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Liu et al., *LLaVA* — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Goodfellow et al., *Generative Adversarial Networks* — [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Ho et al., *Denoising Diffusion Probabilistic Models* — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)* — [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

**Courses / talks**
- Yannic Kilcher's YouTube channel, [@YannicKilcher](https://www.youtube.com/@YannicKilcher) — paper-by-paper walkthroughs of nearly everything above, at the right depth.
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course) — hands-on, code-first path through diffusion models specifically.
- [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) for fast, visual intuition on new generative results as they land.

**Code**
- [OpenAI CLIP repository](https://github.com/openai/CLIP)
- [Hugging Face `diffusers` library](https://github.com/huggingface/diffusers) — the standard toolkit for running and fine-tuning diffusion models.
- [`lightly` self-supervised learning library](https://github.com/lightly-ai/lightly) — clean reference implementations of SimCLR, BYOL, DINO, MAE.

---

[Next: Stage 5 — Specialization & Systems Thinking →](/learn-computer-vision/stage-5-specialization/)
