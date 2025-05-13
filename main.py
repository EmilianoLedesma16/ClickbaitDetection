# main.py

import pandas as pd
from src.preprocessing import clean_text_spacy

# 1. Cargar el dataset
df = pd.read_csv('data/TA1C_dataset_detection_train.csv')

# 2. Aplicar preprocesamiento al texto
df['cleaned_text'] = df['Teaser Text'].apply(clean_text_spacy)

# 3. Guardar el dataset limpio
df.to_csv('data/TA1C_dataset_detection_train_cleaned.csv', index=False)

print("Dataset limpio guardado como 'TA1C_dataset_detection_train_cleaned.csv'")
