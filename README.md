---
title: StyleFusion
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# StyleFusion AI
A Neural Style Transfer application using AdaIN (Adaptive Instance Normalization).

## Live Demo
Try it live on this Hugging Face Space — no setup required.
https://huggingface.co/spaces/Drazan/StyleFusion-HF

---

## Overview

StyleFusion implements the AdaIN architecture introduced by Huang and Belongie (2017) in *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. The model separates content and style representations using a pretrained VGG encoder, aligns the statistical moments of the content feature maps to those of the style image via AdaIN, and reconstructs the output through a learned decoder network.

> ⚠️ **Deployment Note:** The full Flask-based version of this project **cannot be deployed on cloud service providers such as Render, Railway, or Heroku** due to the large memory and compute requirements of the PyTorch model. This Hugging Face Space is the recommended way to run and share the application publicly.

---

## Repository Structure

```
StyleFusion/                    # Core model and application code
│── app.py                      # Flask application entry point
│──utils/                       
│     ├── models.py             # VGGEncoder and Decoder architecture
│     └── utils.py              # AdaIN operation, transforms
├── templates/
│     └── index.html            # Jinja2 frontend template
├── assets/
|     ├── content_image1.jpg
|     ├── style_image1.jpg
|     └── style_image2.jpg
├── weights/
|     ├── decoder_final.pth     # model parameters after training
|     └── vgg_normalised        # pre-trained endoer
|
├── Research Papers Summary/    # Summaries of key NST research papers
|
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## Features
- Upload content + style images
- Adjustable style strength (alpha slider)
- Before/After comparison slider
- Download output image
- Interactive model explanation
- Example gallery

---

## Model Architecture

**Encoder** — A VGG-19 network pretrained on ImageNet, truncated at `relu4_1`, used to extract multi-scale feature representations. Weights are loaded from `vgg_normalised.pth` and kept frozen throughout training.

**AdaIN** — Adaptive Instance Normalization aligns the channel-wise mean and standard deviation of the content features to match those of the style features:

```
AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
```

**Decoder** — A mirror of the encoder (without pooling layers), trained from scratch to invert the AdaIN-transformed feature maps back into pixel space.

**Loss** — Content loss is computed as MSE between the decoded output features and the AdaIN target at `relu4_1`. Style loss is computed as MSE between the channel-wise means and standard deviations of the output and style features across all four VGG relu layers.

![AdaIN Style Transfer Architecture](adain_algo.png)
*Figure: An overview of the style transfer algorithm. A fixed VGG-19 encoder encodes both content and style images. The AdaIN layer aligns feature statistics in latent space. A learned decoder inverts the result back to pixel space. The same VGG encoder is reused to compute content loss (𝓛c) and style loss (𝓛s).*

---

## Training

Training was conducted on Kaggle using a two-phase curriculum to stabilize learning and improve high-resolution output quality.

**Phase 1**
- Epochs: 160
- Image size: 256 x 256
- Style weight: 5
- Batch size: 4

**Phase 2** (resumed from Phase 1 checkpoint)
- Epochs: 100
- Image size: 512 x 512
- Style weight: 10
- Batch size: 4

Both phases use Adam optimization with a learning rate of `1e-4` and a decay schedule of `lr / (1 + decay * epoch)`.

---

## Tech Stack
- PyTorch
- Gradio
- Hugging Face Spaces

---

## Demo Flow
1. Upload a **content image** — the photograph or image whose structure you want to preserve.
2. Upload a **style image** — the artwork whose visual texture and brushwork you want to apply.
3. Adjust the **alpha slider** to control the degree of stylization. A value of `1.0` applies the full style; lower values blend the output toward the original content.
4. Click **Generate**. The stylized image will be displayed.
5. Compare using the Before/After slider and download your result.

---

## Dependencies

| Package | Version |
|---|---|
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| Pillow | 12.0.0 |
| numpy | >=1.24, <2.0 |
| gradio | latest |
| tqdm | 4.66.4 |

---

## Research References

- Gatys et al. (2015) — *A Neural Algorithm of Artistic Style* — the original optimization-based NST approach that this work builds upon.
- Johnson et al. (2016) — *Perceptual Losses for Real-Time Style Transfer* — introduced feed-forward networks for fast per-style transfer.
- Huang & Belongie (2017) — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* — the primary paper implemented in this project.

---

## Author

Devashish Rawat  
[GitHub](https://github.com/Devashish-Rawat1) · [Instagram](https://www.instagram.com/devashish__rawat)

---

## License

This project is released for academic and personal use. The VGG weights used for the encoder are subject to their original license terms from the Visual Geometry Group, University of Oxford.
