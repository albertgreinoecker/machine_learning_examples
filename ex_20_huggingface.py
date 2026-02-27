import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"

'''
Registriere dich auf Hugging Face, erstelle einen API-Token und lege diesen in deiner .env Datei als HF_TOKEN ab.
'''

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json"
}

'''
Werte für den "temperature" Parameter:
0.0	Deterministisch, immer fast gleiche Antwort
0.2	Sehr sachlich, gut für Erklärungen
0.5	Ausgewogen
0.7	Kreativ, aber noch kontrolliert
1.0+	Sehr kreativ, unvorhersehbar
'''

payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [
        {
            "role": "user",
            "content": "Erkläre das Strategy Pattern in Java in Latex code."
        }
    ],
    "max_tokens": 2000,
    "temperature": 0.7
}

response = requests.post(API_URL, headers=headers, json=payload)

print(response.status_code)
print(response.json())

# Ausgabe extrahieren
if response.status_code == 200:
    print(
        response.json()["choices"][0]["message"]["content"]
    )
