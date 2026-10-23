import argparse, json
from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms

MODEL_NAME = "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"
IMG_SIZE = 224

def main():
    parser = argparse.ArgumentParser(description="Prediksi kematangan pisang dengan MobileNetV4 Conv Small")
    parser.add_argument("--image", required=True, help="Path gambar yang akan diprediksi")
    parser.add_argument("--model", default="models/banana_mobilenetv4.pth", help="Path model hasil training")
    parser.add_argument("--classes", default="class_names.json", help="Path class_names.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    classes_path = Path(args.classes)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}. Jalankan training terlebih dahulu.")
    if not image_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Daftar kelas tidak ditemukan: {classes_path}. Jalankan training terlebih dahulu.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    with classes_path.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    with Image.open(image_path) as im:
        x = transform(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    ranking = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
    print(f"Gambar: {image_path}")
    print(f"Prediksi: {ranking[0][0]} ({ranking[0][1]*100:.2f}%)")
    print("Probabilitas setiap kelas:")
    for label, prob in ranking:
        print(f"  {label}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
