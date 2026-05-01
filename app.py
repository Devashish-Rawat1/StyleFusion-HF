import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import tempfile
import time

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# DEVICE (HF free tier = CPU)
device = torch.device("cpu")

# LOAD MODELS
encoder = VGGEncoder("weights/vgg_normalised.pth").to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load("weights/decoder_final.pth", map_location=device)
)

encoder.eval()
decoder.eval()

# STYLE TRANSFER FUNCTION
def stylize(content, style, alpha):
    time.sleep(0.4)

    content_tensor = transforms.ToTensor()(content).unsqueeze(0).to(device)
    style_tensor = transforms.ToTensor()(style).unsqueeze(0).to(device)

    with torch.no_grad():
        c_feat = encoder(content_tensor, is_test=True)
        s_feat = encoder(style_tensor, is_test=True)

        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat

        output = decoder(t)

    output = output.squeeze().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(output)

    # SAVE TEMP FILE FOR DOWNLOAD
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)

    return img, temp_file.name


def compare_fn(content_img, result_img):
    return (content_img, result_img)


# 🎨 MODERN CSS
css = """
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

h1 {
    font-size: 3rem;
    text-align: center;
    background: linear-gradient(90deg, #22d3ee, #f43f5e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}

button {
    background: linear-gradient(135deg, #06b6d4, #22d3ee) !important;
    border-radius: 14px !important;
    font-weight: bold;
    padding: 12px 24px;
}

footer {
    display: none !important;
}

@keyframes pulseGlow {
    0% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.08); }
    100% { opacity: 0.6; transform: scale(1); }
}

.pipeline-step {
    text-align: center;
    font-weight: bold;
    animation: pulseGlow 2s infinite;
}
"""

# EXAMPLES
examples = [
    ["assets/content1.jpg", "assets/style1.jpg", 1.0],
    ["assets/content2.jpg", "assets/style2.jpg", 0.8],
]

# UI
with gr.Blocks(css=css) as demo:

    gr.Markdown("# 🎨 StyleFusion AI")
    gr.Markdown("### Neural Style Transfer using AdaIN")

    with gr.Row():
        content = gr.Image(type="pil", label="Content Image")
        style = gr.Image(type="pil", label="Style Image")

    alpha = gr.Slider(0, 1, value=1.0, label="Style Strength")

    btn = gr.Button("✨ Generate Stylized Image")

    with gr.Row():
        output = gr.Image(label="Output")
        download = gr.File(label="⬇️ Download Image")

    # BEFORE AFTER
    compare = gr.ImageSlider(label="Before vs After")

    btn.click(
        fn=stylize,
        inputs=[content, style, alpha],
        outputs=[output, download]
    ).then(
        fn=compare_fn,
        inputs=[content, output],
        outputs=compare
    )

    # EXAMPLES
    gr.Examples(
        examples=examples,
        inputs=[content, style, alpha],
        label="Try Examples"
    )

    # MODEL EXPLANATION
    gr.Markdown("## 🧠 How It Works")

    with gr.Row():
        gr.Markdown("<div class='pipeline-step'>📷 Content</div>")
        gr.Markdown("<div class='pipeline-step'>➡️</div>")
        gr.Markdown("<div class='pipeline-step'>🧠 Encoder</div>")
        gr.Markdown("<div class='pipeline-step'>➡️</div>")
        gr.Markdown("<div class='pipeline-step'>🎨 AdaIN</div>")
        gr.Markdown("<div class='pipeline-step'>➡️</div>")
        gr.Markdown("<div class='pipeline-step'>🖼 Decoder</div>")
        gr.Markdown("<div class='pipeline-step'>➡️</div>")
        gr.Markdown("<div class='pipeline-step'>✨ Output</div>")

    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using PyTorch + Gradio")

demo.launch()