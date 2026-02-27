import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import os

def download_image(url, save_path="image.jpg"):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    else:
        raise Exception("Failed to download image")

def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, processor

def predict_image(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Hole den index des höchsten Wertes (-1 ist die letzte Dimension)
    predicted_class_idx = logits.argmax(-1).item()
    labels = model.config.id2label
    return labels[predicted_class_idx]

if __name__ == "__main__":
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    #image_url = "https://r-express.com/thumbnail/7b/8e/64/1698073205/14418_Wolfsbarsch-rund-OWN-1_800x800.jpg"
    image_path = download_image(image_url)

    model, processor = load_model()
    # Diese KLassen gibt es:
    for idx, label in model.config.id2label.items():
        print(f"{idx}: {label}")

    prediction = predict_image(image_path, model, processor)
    print(f"Predicted class: {prediction}")
