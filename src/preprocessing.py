import re
import string
import spacy

# Cargar el modelo de lenguaje en inglés
nlp = spacy.load("es_core_news_sm")

def normalize_text(text, mode):
    """
    Normaliza el texto según el modo especificado.
    Modes:
        - "tokenization": Devuelve los tokens del texto.
        - "text_cleaning": Aplica limpieza básica (minúsculas, sin puntuación, sin números).
        - "remove_stopwords": Elimina stopwords.
        - "lemmatization": Aplica lematización.
    """
    if not isinstance(text, str):
        return ""

    # Procesar el texto con spaCy
    doc = nlp(text)

    if mode == "tokenization":
        # Devuelve una lista de tokens
        return [token.text for token in doc]

    elif mode == "text_cleaning":
        # Minúsculas, sin puntuación, sin números
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return text

    elif mode == "remove_stopwords":
        # Devuelve una lista de tokens sin stopwords
        return [token.text for token in doc if not token.is_stop]

    elif mode == "lemmatization":
        # Devuelve una lista de lemas
        return [token.lemma_ for token in doc]

    else:
        raise ValueError("Modo de normalización no válido. Usa 'tokenization', 'text_cleaning', 'remove_stopwords' o 'lemmatization'.")