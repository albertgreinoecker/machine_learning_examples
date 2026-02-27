import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os

# Modell & Prozessor laden
model_name = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

def pdf_to_images(pdf_path):
    """Konvertiert PDF in Bilder (eine Seite pro Bild)"""
    return convert_from_path(pdf_path)

def detect_figures(image):
    """Nutzt LayoutLMv3 zur Layout-Analyse"""
    # Konvertiere Bild zu Tensor
    encoding = processor(images=image, return_tensors="pt")

    # Verhindert "IndexError: index out of range in self"
    if "input_ids" not in encoding or encoding["input_ids"].max() >= model.config.vocab_size:
        print("⚠ Fehler: Ungültige Token-Werte. Skipping...")
        return []

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predicted_tokens = torch.argmax(logits, dim=-1)

    # Klassen abrufen
    id2label = model.config.id2label
    figures = []
    for idx, (bbox, label_id) in enumerate(zip(encoding["bbox"][0], predicted_tokens[0])):
        label = id2label.get(label_id.item(), "unknown")
        if label == "figure":  # Klasse "figure" extrahieren
            figures.append(bbox.tolist())

    return figures

def extract_figures(image, figures, save_dir="extracted_figures/"):
    """Speichert erkannte Abbildungen aus dem Dokument"""
    os.makedirs(save_dir, exist_ok=True)

    for i, bbox in enumerate(figures):
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(f"{save_dir}figure_{i}.jpg")

if __name__ == "__main__":
    pdf_path = "/home/albert/tmp/bierstindl.pdf"  # Dein gescanntes PDF
    images = pdf_to_images(pdf_path)

    for i, image in enumerate(images):
        figures = detect_figures(image)
        if figures:  # Falls Abbildungen erkannt wurden
            extract_figures(image, figures)

    print("✅ Abbildungen erfolgreich extrahiert!")
