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

import random


# Preproceso

# Carga tus datos desde el archivo Excel
data = pd.read_excel("datos_modificados.xlsx")

# Elimina la columna "Unnamed: 0"
data = data.drop("Unnamed: 0", axis=1)

# Selecciona las columnas relevantes para el análisis
# columns = ['Sucursal','UsuarioEntrada', 'HoraEntrada', 'below_5', 'between_5_and_30', 'between_30_and_60', 'between_60_and_90', 'above_90', 'Month', 'MonthDay', 'WeekDay', 'transaction_1', 'transaction_2', 'transaction_3']
columns = ['Sucursal', 'HoraEntrada', 'below_5', 'between_5_and_30', 'between_30_and_60',
           'between_60_and_90', 'above_90', 'transaction_1', 'transaction_2', 'transaction_3']

# Crea un nuevo DataFrame solo con las columnas seleccionadas
selected_data = data[columns]

# Normaliza los datos utilizando la norma L2
normalized_data = normalize(selected_data, norm='l2')

# Definir los pesos para las columnas ponderadas
peso_below_5 = -2.5
peso_transaction_1 = -2.5
peso_transaction_3 = -2.5

# Aplica los castigos a las columnas normalizadas
normalized_data[:, 3] *= peso_below_5  # Ponderación para 'below_5'
normalized_data[:, 7] *= peso_transaction_1  # Ponderación para 'transaction_1'
normalized_data[:, 9] *= peso_transaction_3  # Ponderación para 'transaction_3'

if 'normalized_data' not in st.session_state:
    st.session_state.normalized_data = normalized_data

# Configuración de la página
st.title('Cargando modelos')
st.header('Isolation Forest')


with st.spinner("Corriendo Isolation Forest."):
    time.sleep(0.1)
    dataIF = data.copy(deep=copy)

    # Crea una instancia del modelo Isolation Forest
    model = IsolationForest(random_state=42)

    # Ajusta el modelo a tus datos normalizados y con los castigos aplicados
    model.fit(normalized_data)

    # Obtén los puntajes de anomalía para cada muestra
    anomaly_scores = model.decision_function(normalized_data)

    # Establece el umbral
    umbral = np.percentile(anomaly_scores, 5)

    # Clasifica las muestras según el umbral y los puntajes de anomalía
    etiquetas = ['Normal' if score >= umbral else 'Anormal' for score in anomaly_scores]

    # Agrega las etiquetas y los puntajes de anomalía como columnas adicionales en tus datos originales
    dataIF['Anomaly Score'] = anomaly_scores
    dataIF['Etiqueta'] = etiquetas

    # Guarda los resultados en un nuevo archivo Excel
    dataIF.to_excel("resultados_IsolationF.xlsx", index=False)

# Después de completar el entrenamiento:
st.success('Listo!')

data1 = pd.read_excel("resultados_IsolationF.xlsx")
filtered_df = data1[(data1['Sucursal'] == 2466)]
chart_data = pd.DataFrame(filtered_df, columns=['HoraEntrada', 'Anomaly Score', 'Etiqueta'])

print(chart_data)


st.vega_lite_chart(chart_data, {
    'mark': {'type': 'circle', 'tooltip': True},
    'encoding': {
        'x': {'field': 'HoraEntrada', 'type': 'quantitative'},
        'y': {'field': 'Anomaly Score', 'type': 'quantitative'},
        'size': {'field': 'Etiqueta', 'type': 'nominal'},
        'color': {'field': 'Etiqueta', 'type': 'nominal'},
    },
})
