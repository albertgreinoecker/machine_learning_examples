from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Modell und Tokenizer laden
model_name = "facebook/blenderbot-400M-distill"  #Knowledge Distillation
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)


# Chat-Funktion
def chat():
    print("Chatbot gestartet! Tippe 'exit', um den Chat zu beenden.")
    while True:
        user_input = input("Du: ")
        if user_input.lower() == "exit":
            print("Chatbot beendet!")
            break

        # Eingabe tokenisieren
        inputs = tokenizer(user_input, return_tensors="pt")

        # Antwort generieren
        response_ids = model.generate(**inputs)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        print(f"Bot: {response}")


if __name__ == "__main__":
    chat()
