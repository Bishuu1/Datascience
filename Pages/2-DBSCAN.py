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


# Configuración de la página
st.title('Cargando modelos')
st.header('DBSCAN')


with st.spinner("Corriendo DBSCAN."):
    time.sleep(0.1)

    # Cargar los datos desde el archivo Excel
    # data_dbscan = data.copy(deep=copy)
    data_dbscan = pd.read_excel('datos_modificados.xlsx')
    # Preprocesar los datos
    # Aquí debes realizar cualquier preprocesamiento necesario, como normalización de los datos, codificación de variables categóricas, etc.
    # Asegúrate de que los datos estén en el formato adecuado para el entrenamiento del modelo.

    columns_to_drop = ['Sucursal', 'UsuarioEntrada', 'Month', 'MonthDay', 'WeekDay']

    # Obtener una copia de los datos sin las columnas eliminadas
    data_processed = data_dbscan.drop(columns_to_drop, axis=1)

    # Separar los datos en conjunto de entrenamiento
    train_data = data_processed.values

    # Obtener el índice de las filas
    filas = np.arange(train_data.shape[0])

    # Mezclar aleatoriamente las filas
    np.random.shuffle(filas)

    # Aplicar el shuffle a los datos
    train_data = train_data[filas]

    # Definir los parámetros para DBSCAN
    eps = 7.4  # Ajusta el valor del radio
    min_samples = 4  # Ajusta el número mínimo de muestras para formar un grupo

    # Crear y entrenar el modelo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(train_data)

    # Obtener las etiquetas de los grupos (clusters)
    labels = dbscan.labels_

    # Identificar las transacciones sospechosas (fraudulentas)
    suspected_frauds = np.where(labels == -1)[0]

    # Calcular la distancia al punto de referencia más cercano
    nbrs = NearestNeighbors(n_neighbors=2).fit(train_data)
    distances, indices = nbrs.kneighbors(train_data)

    distances_to_reference = distances[:, 1]  # Distancia al punto de referencia más cercano

    # Ordenar los casos anómalos por distancia descendente
    anomalous_cases = sorted(zip(suspected_frauds, distances_to_reference[suspected_frauds]),
                             key=lambda x: x[1], reverse=True)

    # Seleccionar los 138 casos más anómalos
    top_anomalous_cases = anomalous_cases[:138]

    # Agregar la columna 'Etiqueta' al DataFrame original
    data_dbscan['Etiqueta'] = 'Normal'

    # Marcar los 138 casos más anómalos
    for case in top_anomalous_cases:
        data_dbscan.loc[case[0], 'Etiqueta'] = 'Anormal'

    # Imprimir los 138 casos más anómalos
    print("Los 138 casos más anómalos:")
    print(data_dbscan[data_dbscan['Etiqueta'] == 'Anormal'])

    # Guardar los datos actualizados en un archivo Excel
    data_dbscan.to_excel('resultados_dbscan.xlsx', index=False)

# Después de completar el entrenamiento:
st.success('Listo!')

