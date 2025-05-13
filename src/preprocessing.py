# src/preprocessing.py

import re
import string
import spacy

# Carga del modelo de lenguaje en inglés
# Asegúrate de haber hecho: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def clean_text_spacy(text, lowercase=True, remove_punct=True, remove_digits=True,
                     remove_stopwords=True, lemmatize=True):
    if not isinstance(text, str):
        return ""
    
    # Paso 1: minúsculas
    if lowercase:
        text = text.lower()
    
    # Paso 2: quitar signos de puntuación
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Paso 3: quitar números
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Paso 4: procesamiento con spaCy
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        if lemmatize:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)

    # Devuelve el texto limpio
    return ' '.join(tokens)
