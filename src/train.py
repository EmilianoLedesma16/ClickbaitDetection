import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

def train_model(train_path, dev_path, model_output_path):
    # Cargar los datasets
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    # Crear representaciones de texto (TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigramas y bigramas
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_dev = vectorizer.transform(dev_df['cleaned_text'])
    y_train = train_df['Tag Value']
    y_dev = dev_df['Tag Value']

    # Entrenar un modelo de regresión logística
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_dev)
    print("Reporte de clasificación en el conjunto de desarrollo:")
    print(classification_report(y_dev, y_pred))

    # Guardar el modelo y el vectorizador
    dump((model, vectorizer), model_output_path)
    print(f"Modelo guardado en {model_output_path}")

# Ejecutar el entrenamiento si este archivo se ejecuta directamente
if __name__ == "__main__":
    train_model(
        train_path="../data/TA1C_dataset_detection_train_split.csv",
        dev_path="../data/TA1C_dataset_detection_dev_split.csv",
        model_output_path="../models/logistic_regression_model.joblib"
    )