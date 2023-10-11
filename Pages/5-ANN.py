import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import datetime
import os
import re
import time
from pathlib import Path
from streamlit_extras.switch_page_button import switch_page
from sklearn.ensemble import IsolationForest
import io
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense

# Configuración de la página
st.title('Cargando modelos')
st.header('Red Neuronal')


with st.spinner("Corriendo ANN."):
    time.sleep(0.1)

    # Leer el conjunto de datos existente
    df = pd.read_excel('resultados_cruce.xlsx', index_col=0)

    # Eliminar columnas innecesarias
    columnas_a_dropear = ['Etiqueta_IsolationF', 'Etiqueta_LOF', 'Etiqueta_autoencoder', 'Etiqueta_dbscan']
    df = df.drop(columnas_a_dropear, axis=1)

    # Verificar y reemplazar valores no deseados en la columna 'Fraude'
    reemplazos = {
        'Normal': 'No fraud',
        'Anormal': 'Anomaly',
        'Muy Anormal': 'high Anomaly'
    }

    df['Fraude'] = df['Fraude'].replace(reemplazos)

    # Dividir los datos en características (X) y etiquetas (y)
    X = df.drop('Fraude', axis=1)
    y = df['Fraude'].replace(['No fraud', 'Anomaly', 'high Anomaly'], [0, 1, 2]).astype(int)

    # Realizar cualquier transformación necesaria en los datos, como codificación de variables categóricas

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = Sequential()
    # Agregar capas y configurar el modelo
    model.add(Dense(32, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Almacenar el historial de entrenamiento para graficar las métricas
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)

    # Obtener las etiquetas predichas en el conjunto de prueba
    predicted_labels = predictions.argmax(axis=1)
    true_labels = y_test.values

    # Imprimir las etiquetas reales y predichas en el conjunto de prueba
    for i in range(len(true_labels)):
        print(f"Dato de prueba {i+1}: Etiqueta real: {true_labels[i]}, Etiqueta predicha: {predicted_labels[i]}")

    # Calcular métricas de evaluación en el conjunto de prueba
    precision = precision_score(y_test, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(y_test, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(y_test, predicted_labels, average='weighted', zero_division=1)

    print(f'Precisión en el conjunto de prueba: {precision:.4f}')
    print(f'Recall en el conjunto de prueba: {recall:.4f}')
    print(f'F1-score en el conjunto de prueba: {f1:.4f}')

    # Graficar la precisión, recall y F1-score a través de las épocas
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)


# Después de completar el entrenamiento:
st.success('Listo!')

