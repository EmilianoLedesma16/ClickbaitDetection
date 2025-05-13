# src/test_preprocessing.py

from preprocessing import clean_text_spacy

# Prueba con un texto ejemplo
sample_text = "5 Ways You Can Instantly Improve Your Productivity Today!"

# Limpiar el texto usando la función de spaCy
cleaned_text = clean_text_spacy(sample_text)

print("Texto original:")
print(sample_text)

print("\nTexto limpio:")
print(cleaned_text)
