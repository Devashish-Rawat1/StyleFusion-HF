import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import tempfile
import time

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# ─── DEVICE ───────────────────────────────────────────────────────────────────
device = torch.device("cpu")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
encoder = VGGEncoder("weights/vgg_normalised.pth").to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("weights/decoder_final.pth", map_location=device))
encoder.eval()
decoder.eval()

# ─── STYLE TRANSFER ───────────────────────────────────────────────────────────
def stylize(content, style, alpha, progress=gr.Progress(track_tqdm=True)):
    if content is None or style is None:
        raise gr.Error("Please upload both a content image and a style image.")

    progress(0, desc="Preprocessing images…")
    content_tensor = transforms.ToTensor()(content).unsqueeze(0).to(device)
    style_tensor   = transforms.ToTensor()(style).unsqueeze(0).to(device)

    progress(0.3, desc="Encoding features…")
    with torch.no_grad():
        c_feat = encoder(content_tensor, is_test=True)
        s_feat = encoder(style_tensor,   is_test=True)

        progress(0.6, desc="Applying AdaIN…")
        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat

        progress(0.85, desc="Decoding stylized image…")
        output = decoder(t)

    output = output.squeeze().cpu().clamp(0, 1)
    img    = transforms.ToPILImage()(output)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)

    progress(1.0, desc="Done!")
    return img, temp_file.name


def compare_fn(content_img, result_img):
    return (content_img, result_img)


# ─── THEME ────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.rose,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Sora"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    block_label_text_color="*neutral_300",
    input_background_fill="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg,#06b6d4,#0ea5e9)",
    button_primary_background_fill_hover="linear-gradient(135deg,#0ea5e9,#22d3ee)",
    button_primary_text_color="#ffffff",
    slider_color="*primary_500",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
css = """
/* ── GLOBAL ── */
:root {
    --cyan:   #22d3ee;
    --rose:   #f43f5e;
    --gold:   #fbbf24;
    --glass:  rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --radius: 18px;
}

* { box-sizing: border-box; }

body {
    background: #020617 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(6,182,212,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%,  rgba(244,63,94,0.12) 0%, transparent 60%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 16px 60px !important;
}

footer { display: none !important; }

/* ── HERO HEADER ── */
.hero-wrap {
    text-align: center;
    padding: 56px 0 32px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.3);
    color: var(--cyan);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4.2rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, var(--cyan) 50%, var(--rose) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 14px;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto 10px;
    line-height: 1.7;
}

/* ── STAT CHIPS ── */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 24px 0 0;
}
.stat-chip {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 7px 16px;
    font-size: 0.8rem;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 7px;
    backdrop-filter: blur(8px);
}
.stat-chip span { color: var(--cyan); font-weight: 700; }

/* ── CARDS ── */
.card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    backdrop-filter: blur(12px);
    transition: border-color 0.25s, box-shadow 0.25s;
}
.card:hover {
    border-color: rgba(34,211,238,0.25);
    box-shadow: 0 0 32px rgba(34,211,238,0.07);
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── IMAGE PANELS ── */
.image-panel label {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
}
.image-panel .svelte-1oiin9d {  /* upload area */
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.02) !important;
    transition: border-color 0.2s !important;
}
.image-panel .svelte-1oiin9d:hover {
    border-color: rgba(34,211,238,0.4) !important;
}

/* ── ALPHA SLIDER ── */
.alpha-wrap {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
}
.alpha-wrap label { 
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}
.alpha-wrap input[type=range]::-webkit-slider-thumb {
    background: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(34,211,238,0.5) !important;
}

/* ── GENERATE BUTTON ── */
.gen-btn {
    background: linear-gradient(135deg,#06b6d4,#0ea5e9) !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    height: 56px !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 24px rgba(6,182,212,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100% !important;
}
.gen-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(6,182,212,0.45) !important;
}
.gen-btn:active { transform: translateY(0) !important; }

/* ── OUTPUT AREA ── */
.output-panel label {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* ── PIPELINE ── */
.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    margin: 8px 0 4px;
}
.pipe-node {
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 10px;
    padding: 9px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
    text-align: center;
    min-width: 80px;
}
.pipe-node .pipe-icon { font-size: 1.1rem; display: block; margin-bottom: 2px; }
.pipe-arrow {
    color: #334155;
    font-size: 1.2rem;
    font-weight: 300;
}

/* ── TECH BADGES ── */
.tech-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.tech-badge {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.05em;
}

/* ── TIPS ACCORDION ── */
.tips-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 0.87rem;
    color: #d1d5db;
    line-height: 1.8;
}
.tips-box strong { color: var(--gold); }

/* ── ADAIN EXPLAINER ── */
.adain-box {
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 14px;
    padding: 20px 24px;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.8;
}
.adain-box code {
    background: rgba(34,211,238,0.12);
    border-radius: 5px;
    padding: 2px 7px;
    color: var(--cyan);
    font-size: 0.82rem;
}

/* ── EXAMPLES LABEL ── */
.examples-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin: 32px 0 12px;
}

/* ── DIVIDER ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 40px 0 32px;
}

/* ── FOOTER ── */
.site-footer {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    margin-top: 48px;
    line-height: 2;
}
.site-footer a { color: #475569; text-decoration: none; }
.site-footer a:hover { color: var(--cyan); }
"""

# ─── EXAMPLE PAIRS ────────────────────────────────────────────────────────────
examples = [
    ["assets/content_image1.jpg", "assets/style_image1.jpg", 1.0],
    ["assets/content_image1.jpg", "assets/style_image2.jpg", 1.0],
]

# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="StyleFusion AI") as demo:

    # ── HERO ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-wrap">
      <div class="hero-badge">🎨 Neural Style Transfer · AdaIN Architecture</div>
      <h1 class="hero-title">StyleFusion AI</h1>
      <p class="hero-sub">
        Reimagine any photograph through the lens of great art. Powered by a
        VGG-based encoder and Adaptive Instance Normalization.
      </p>
      <div class="stat-row">
        <div class="stat-chip">⚡ <span>Real-time</span> inference</div>
        <div class="stat-chip">🖼 <span>Any-resolution</span> images</div>
        <div class="stat-chip">🎛 <span>Tunable</span> style strength</div>
        <div class="stat-chip">💾 <span>PNG</span> download</div>
      </div>
    </div>
    """)

    # ── INPUT SECTION ─────────────────────────────────────────────────────────
    gr.HTML('<p class="section-label">01 — Upload Images</p>')

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="image-panel"):
            content = gr.Image(
                type="pil",
                label="📷  Content Image",
                show_label=True,
                height=320,
            )
        with gr.Column(elem_classes="image-panel"):
            style = gr.Image(
                type="pil",
                label="🎨  Style Image",
                show_label=True,
                height=320,
            )

    # ── CONTROLS ──────────────────────────────────────────────────────────────
    gr.HTML('<p class="section-label" style="margin-top:24px">02 — Configure</p>')

    with gr.Group(elem_classes="alpha-wrap"):
        alpha = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.05,
            label="Style Strength  (α)",
            info="0 = pure content structure · 1 = full style transfer",
        )

    gr.HTML('<div style="height:16px"></div>')
    btn = gr.Button("✨  Generate Stylized Image", elem_classes="gen-btn", variant="primary")

    # ── OUTPUT SECTION ────────────────────────────────────────────────────────
    gr.HTML('<p class="section-label" style="margin-top:32px">03 — Results</p>')

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="output-panel"):
            output = gr.Image(label="🖼  Stylized Output", show_label=True, height=380)
        with gr.Column(elem_classes="output-panel"):
            download = gr.File(label="⬇️  Download PNG", show_label=True)

    # ── BEFORE / AFTER SLIDER ─────────────────────────────────────────────────
    gr.HTML('<p class="section-label" style="margin-top:32px">04 — Before vs After</p>')
    compare = gr.ImageSlider(
        label="Drag to compare content ↔ stylized",
        show_label=True,
        height=380,
    )

    # ── EXAMPLES ──────────────────────────────────────────────────────────────
    gr.HTML('<p class="examples-header">✦ Try a Preset Pair</p>')
    gr.Examples(
        examples=examples,
        inputs=[content, style, alpha],
        outputs=[output, download],
        fn=stylize,
        cache_examples="lazy",
        label="",
        examples_per_page=4,
    )

    # ── TIPS ACCORDION ────────────────────────────────────────────────────────
    gr.HTML('<hr class="divider">')
    with gr.Accordion("💡  Tips for Best Results", open=False):
        gr.HTML("""
        <div class="tips-box">
          <strong>Content image:</strong> Use high-resolution photos (portraits, landscapes, architecture) with clear subjects.<br>
          <strong>Style image:</strong> Paintings with strong texture and color work best — try Van Gogh, Monet, or Kandinsky.<br>
          <strong>α = 0.3–0.6:</strong> Subtle stylization that preserves natural colours.<br>
          <strong>α = 0.8–1.0:</strong> Full artistic transformation for maximum dramatic effect.<br>
          <strong>Aspect ratio:</strong> Square-ish images (512 × 512) produce the most balanced results.
        </div>
        """)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    with gr.Accordion("🧠  How It Works — AdaIN Architecture", open=False):
        gr.HTML("""
        <div class="pipeline" style="margin-bottom:20px">
          <div class="pipe-node"><span class="pipe-icon">📷</span>Content</div>
          <span class="pipe-arrow">→</span>
          <div class="pipe-node"><span class="pipe-icon">🧠</span>VGG Enc.</div>
          <span class="pipe-arrow">→</span>
          <div class="pipe-node"><span class="pipe-icon">⚡</span>AdaIN</div>
          <span class="pipe-arrow">→</span>
          <div class="pipe-node"><span class="pipe-icon">🔧</span>Decoder</div>
          <span class="pipe-arrow">→</span>
          <div class="pipe-node"><span class="pipe-icon">✨</span>Output</div>
          <br>
          <div class="pipe-node" style="margin-top:8px"><span class="pipe-icon">🎨</span>Style</div>
        </div>
        <div class="adain-box">
          <strong style="color:#e2e8f0">Adaptive Instance Normalization (AdaIN)</strong><br><br>
          The core operation is elegantly simple:<br><br>
          <code>AdaIN(x, y) = σ(y) · ( (x − μ(x)) / σ(x) ) + μ(y)</code><br><br>
          The <strong style="color:#e2e8f0">VGG encoder</strong> maps both images to a shared feature space.
          AdaIN then aligns the <em>mean</em> and <em>standard deviation</em> of the content features
          to match those of the style features — transferring texture and colour statistics without
          any per-style optimization loop. The lightweight <strong style="color:#e2e8f0">decoder</strong>
          reconstructs a full image from the adapted features in a single forward pass,
          enabling real-time stylization.
        </div>
        """)
        gr.HTML("""
        <div class="tech-row" style="margin-top:16px">
          <span class="tech-badge">PyTorch</span>
          <span class="tech-badge">VGG-19</span>
          <span class="tech-badge">AdaIN</span>
          <span class="tech-badge">Gradio 4</span>
          <span class="tech-badge">HuggingFace Spaces</span>
          <span class="tech-badge">Python 3.10</span>
        </div>
        """)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="site-footer">
      Built with PyTorch + Gradio &nbsp;·&nbsp;
      Paper: <a href="https://arxiv.org/abs/1703.06868" target="_blank">Huang & Belongie 2017</a> &nbsp;·&nbsp;
      <a href="https://huggingface.co" target="_blank">HuggingFace Spaces</a>
    </div>
    """)

    # ── EVENTS ────────────────────────────────────────────────────────────────
    btn.click(
        fn=stylize,
        inputs=[content, style, alpha],
        outputs=[output, download],
    ).then(
        fn=compare_fn,
        inputs=[content, output],
        outputs=compare,
    )

demo.launch(theme=theme, css=css)