import tiktoken

# GPT-4 Encoding laden (funktioniert auch für GPT-3.5)
encoding = tiktoken.encoding_for_model("gpt-4")

# Beispieltext
text = "Die Reise zum Mars beginnt! Wir starten in 3, 2, 1..."

# Text in Tokens umwandeln
tokens = encoding.encode(text)

# Tokens ausgeben
print("Token IDs:", tokens)
print("Anzahl der Tokens:", len(tokens))


# Tokens in lesbaren Text umwandeln
decoded_tokens = [encoding.decode([t]) for t in tokens]
print("Decodierte Tokens:", decoded_tokens)
