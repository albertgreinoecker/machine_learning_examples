from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, SnowballStemmer
import string

# Falls notwendig, lade die benötigten Ressourcen herunter
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger_de') #Part-of-Speech-Tagger
#nltk.download('stopwords')

#nltk.download() # Öffnet einen Download-Dialog

# Beispieltext
text = "Was gibt es besseres als Sprachverarbeitung mit NLTK zu machen."

# Satz-Tokenisierung
sentences = sent_tokenize(text)
print("Sätze:", sentences)

# Wort-Tokenisierung
words = word_tokenize(text)
print("Wörter:", words)

# Stemming (Reduktion auf Wortstamm)
stemmer = SnowballStemmer('german') # PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print("Stemming:", stems)

# Entfernen von Stoppwörtern
stop_words = set(stopwords.words("german"))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Ohne Stoppwörter:", filtered_words)

# Entfernen von Satzzeichen -  Funktion nicht von nltk
clean_tokens = [word for word in words if word not in string.punctuation]
print("Ohne Satzzeichen:", clean_tokens)

# Wortarten-Tagging
pos_tags = pos_tag(words)
print("Wortarten-Tagging:", pos_tags)
