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


# df = pd.read_excel('datos_modificados.xlsx')
# df.head()
# columns_to_drop = ['Sucursal', 'Month', 'WeekDay']
# df = df.drop(columns_to_drop, axis=1)

# Configuración de la página
st.title('Cargando modelos')
st.header('Isolation Forest')


with st.spinner("Corriendo LOF."):
    time.sleep(0.1)
    X_train, X_test = train_test_split(st.session_state.normalized_data, test_size=0.2, random_state=42)
    # Crear una instancia del modelo LOF
    lof = LocalOutlierFactor(
        n_neighbors=10, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=0.05,
        novelty=False)  # Número de vecinos considerados para la detección de anomalías
    # Entrenar el modelo con tus datos
    lof.fit(X_train)
    anomaly_scores = lof.negative_outlier_factor_
    # anomaly_in_test=lof.fit_predict(X_train)
    anomaly_labels = lof.fit_predict(st.session_state.normalized_data)
    len(anomaly_labels)
    df = pd.read_excel('datos_modificados.xlsx')
    df['anomalia'] = pd.Series(anomaly_labels, index=df.index)
    mapping = {-1: 'anormal', 1: 'normal'}
    # Replace the values in the 'status' column using the mapping dictionary
    df['anomalia'] = df['anomalia'].replace(mapping)
    df.to_excel('resultados_lof.xlsx', index=False)
    anomaly_in_test = lof.fit_predict(X_test)

    num_anomalies = len(anomaly_labels[anomaly_labels == -1])

    # Calcular el porcentaje de anomalías
    percentage_anomalies = (num_anomalies / len(anomaly_labels)) * 100

    print(f"Número de anomalías detectadas: {num_anomalies}")
    print(f"Porcentaje de anomalías: {percentage_anomalies:.2f}%")


# Después de completar el entrenamiento:
st.success('Listo!')
