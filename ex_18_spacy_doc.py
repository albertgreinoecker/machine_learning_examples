# python -m spacy download de_core_news_sm es gibt auch sm, md, lg (small, medium, large)
import spacy
import de_core_news_md
import pathlib
from collections import Counter

nlp = de_core_news_md.load()

print(nlp.Defaults.stop_words)

# Stopwort hinzufügen
nlp.Defaults.stop_words.add("der")

doc = nlp("Dies ist ein einfachr Satz um zu zeigen wie gleich mehrere Token verarbeitet werden.")

for token in doc:
    print(token.text, token.pos_, token.dep_)

doc2 = nlp(pathlib.Path('data/die_verwandlung_de.txt').read_text(encoding="utf-8"))
for token in doc2:
    print(token.text, token.pos_, token.dep_, spacy.explain(token.tag_),  token.lemma_)  #dep: syntaktische Abhängigkeitsbeziehungen


# Sentence detection
for s in doc2.sents:
    print("=>", s)

most_common = Counter([token.lemma_ for token in doc2 if not token.is_punct and not token.is_stop and not token.is_space]).most_common(10)
print(most_common)

# Named Entity Recognition
for ent in doc2.ents:
    print(ent.text, ent.label_)