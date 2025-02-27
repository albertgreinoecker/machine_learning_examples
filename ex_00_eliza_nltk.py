from nltk.chat.eliza import eliza_chatbot

print("ELIZA: Hallo! Wie kann ich dir helfen? (Tippe 'exit' zum Beenden)")
print("Das ist allerdings die NLTK-Version von ELIZA, nicht die originale.")
print("Funktioniert nur in Englisch")
while True:
    user_input = input("DU: ")
    if user_input.lower() in ["exit", "bye", "tschüss", "auf wiedersehen"]:
        print("ELIZA: Auf Wiedersehen!")
        break
    print("ELIZA:", eliza_chatbot.respond(user_input))
