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

    # Crear y entrenar el modelo
    model = Sequential()
    # Agregar capas y configurar el modelo
    model.add(Dense(32, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Almacenar el historial de entrenamiento para graficar las métricas
    history = model.fit(X, y, epochs=10, batch_size=32)

    # Hacer predicciones en nuevos datos
    new_data = {
        'Sucursal': [2466, 2466, 2466],
        'UsuarioEntrada': [6918, 6918, 6918],
        'HoraEntrada': [9, 10, 11],
        'below_5': [9, 10, 11],
        'between_5_and_30': [6, 3, 1],
        'between_30_and_60': [9, 12, 6],
        'between_60_and_90': [4, 3, 8],
        'above_90': [2, 2, 1],
        'Month': [3, 3, 3],
        'MonthDay': [1, 1, 1],
        'WeekDay': [9, 9, 9],
        'transaction_1': [0, 0, 0],
        'transaction_2': [0, 0, 0],
        'transaction_3': [0, 0, 0]
    }

    new_data_df = pd.DataFrame(new_data)

    # Realizar las transformaciones necesarias en los datos nuevos, como codificar las variables categóricas si es necesario.

    predictions = model.predict(new_data_df)

    predicted_labels = [label_mapping[idx] for idx in predictions.argmax(axis=1)]

    # Imprimir las etiquetas predichas
    for i, label in enumerate(predicted_labels):
        print(f"Dato de entrada {i+1}: Etiqueta predicha: {label}")

    # Calcular métricas de evaluación
    precision = precision_score(y, model.predict(X).argmax(axis=1), average='weighted')
    recall = recall_score(y, model.predict(X).argmax(axis=1), average='weighted')
    f1 = f1_score(y, model.predict(X).argmax(axis=1), average='weighted')

    print(f'Precisión: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Graficar la precisión, recall y F1-score a través de las épocas
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(accuracy) + 1)
    print(predictions)

# Después de completar el entrenamiento:
st.success('Listo!')
