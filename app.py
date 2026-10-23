from pathlib import Path
import base64
import io

import timm
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Maksimal 8 MB per gambar

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "banana_mobilenetv4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Statistik normalisasi ImageNet yang digunakan saat training.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Pastikan file models/banana_mobilenetv4.pth tersedia."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    classes = checkpoint.get("class_names")
    if not classes:
        raise ValueError("Checkpoint tidak memiliki informasi class_names.")

    # Hilangkan prefix hf_hub: agar arsitektur dibuat dari konfigurasi lokal timm.
    saved_name = checkpoint.get(
        "model_name",
        "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"
    )
    model_name = saved_name.replace("hf_hub:", "hf-hub:")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, classes

MODEL, CLASS_NAMES = load_model()

def predict_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    confidence = float(probs[idx].item()) * 100
    results = sorted(
        [
            {"name": name.replace("_", " "), "confidence": float(probs[i].item()) * 100}
            for i, name in enumerate(CLASS_NAMES)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    return CLASS_NAMES[idx].replace("_", " "), confidence, results

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    results = None
    image_data = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Pilih gambar pisang terlebih dahulu."
        else:
            try:
                raw = uploaded.read()
                image = Image.open(io.BytesIO(raw))
                prediction, confidence, results = predict_image(image)
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((700, 500))
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG", quality=88)
                image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
            except (UnidentifiedImageError, OSError):
                error = "File tidak terbaca sebagai gambar. Gunakan JPG, PNG, atau WEBP."
            except Exception as exc:
                app.logger.exception("Gagal melakukan prediksi")
                error = f"Prediksi gagal: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        results=results,
        image_data=image_data,
        error=error,
        device=str(DEVICE),
    )

@app.errorhandler(413)
def too_large(_error):
    return "Ukuran gambar terlalu besar. Maksimal 8 MB.", 413

if __name__ == "__main__":
    # Buka http://127.0.0.1:5000 di browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
