# python -m spacy download de_core_news_sm es gibt auch sm, md, lg (small, medium, large)
import spacy
import de_core_news_md
import pathlib
from collections import Counter

nlp = de_core_news_md.load()

# Beispielwörter
words = ["Mann", "Frau", "Kinder", "Autos", "Katze", "Kater", "Hund", "Haus" ]

for word in words:
    doc = nlp(word)
    for token in doc:
        print(f"Wort: {token.text}, Geschlecht: {token.morph.get('Gender')}, Number: {token.morph.get('Number')}")

