import re
import random

# Regelbasiertes Antwortsystem mit Spiegelung
reflections = {
    "ich": "du",
    "mir": "dir",
    "mein": "dein",
    "meine": "deine",
    "mich": "dich",
    "bin": "bist",
    "war": "warst",
    "mache": "machst",
    "will": "willst",
    "habe": "hast",
    "kann": "kannst"
}

# Musterbasierte Antwortregeln
patterns = [
    (r"ich brauche (.*)", ["Warum brauchst du {0}?", "Glaubst du, dass {0} dir helfen kann?"]),
    (r"ich fühle mich (.*)", ["Warum fühlst du dich {0}?", "Seit wann fühlst du dich {0}?"]),
    (r"ich (.*)", ["Warum {0} du?", "Wie lange {0} du schon?", "Denkst du oft darüber nach, {0}?"]),
    (r"mein (.*)", ["Erzähl mir mehr über dein {0}.", "Warum erwähnst du dein {0}?"]),
    (r"warum (.*)", ["Was denkst du selbst darüber?", "Vielleicht liegt die Antwort in dir selbst."]),
    (r"ja", ["Warum sagst du 'ja'?", "Erzähl mir mehr darüber."]),
    (r"nein", ["Warum nicht?", "Bist du dir sicher?"]),
    (r"(.*)", ["Erzähl mir mehr darüber.", "Warum denkst du so?", "Interessant, kannst du das erklären?"])
]

def reflektiere(satz):
    """Spiegelt bestimmte Wörter, um personalisierte Antworten zu erzeugen."""
    worte = satz.lower().split()
    gespiegelt = [reflections.get(wort, wort) for wort in worte]
    return " ".join(gespiegelt)

def eliza_antwort(eingabe):
    """Vergleicht die Eingabe mit den Mustern und gibt eine passende Antwort aus."""
    for muster, antworten in patterns:
        match = re.match(muster, eingabe.lower())
        if match:
            antwort = random.choice(antworten)
            return antwort.format(*[reflektiere(gruppe) for gruppe in match.groups()])
    return "Erzähl mir mehr darüber."

# Startet den Chat
print("ELIZA: Hallo! Wie kann ich dir helfen? (Tippe 'exit' zum Beenden)")
while True:
    user_input = input("DU: ")
    if user_input.lower() in ["exit", "bye", "tschüss", "auf wiedersehen"]:
        print("ELIZA: Auf Wiedersehen! Pass auf dich auf.")
        break
    print("ELIZA:", eliza_antwort(user_input))
