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
from tensorflow.keras import layers, models
import random
from keras.models import Sequential
from keras.layers import Dense


#Configuración de la página
st.title('Cargando modelos')
st.header('Local Outliers Factor')


with st.spinner("Corriendo LOF."):
    time.sleep(0.1)
    # Cargar los datos desde el archivo Excel
    data = pd.read_excel('datos_modificados.xlsx')

    # Preprocesar los datos
    # Aquí debes realizar cualquier preprocesamiento necesario, como normalización de los datos, codificación de variables categóricas, etc.
    # Asegúrate de que los datos estén en el formato adecuado para el entrenamiento del modelo.

    columns_to_drop = [ 'UsuarioEntrada', 'Month', 'MonthDay', 'WeekDay']

    # Obtener una copia de los datos sin las columnas eliminadas
    data_processed = data.drop(columns_to_drop, axis=1)

    # Separar los datos en conjunto de entrenamiento
    train_data = data_processed.values

    # Obtener el índice de las filas
    filas = np.arange(train_data.shape[0])

    # Mezclar aleatoriamente las filas
    np.random.shuffle(filas)

    # Aplicar el shuffle a los datos
    train_data = train_data[filas]

    # Construir el modelo de autoencoder
    input_dim = train_data.shape[1]

    # Definir la arquitectura del autoencoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    # Crear el modelo
    autoencoder = models.Model(input_layer, decoded)

    # Compilar el modelo
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True)

    # Obtener las predicciones de reconstrucción
    reconstructions = autoencoder.predict(train_data)

    # Calcular el error de reconstrucción (MSE)
    mse = np.mean(np.power(train_data - reconstructions, 2), axis=1)

    # Establecer un umbral para detectar anomalías
    threshold = np.percentile(mse, 95)  # Por ejemplo, tomar el percentil 95

    # Identificar las transacciones sospechosas (fraudulentas)
    suspected_frauds = np.where(mse > threshold)[0]

    suspected_df = data.iloc[suspected_frauds]
    # Agregar la columna 'Etiqueta' al DataFrame original
    data['Etiqueta'] = 'Normal'
    data.loc[suspected_frauds, 'Etiqueta'] = 'Anormal'

    # Guardar los resultados en un archivo Excel
    data.to_excel('resultados_autoencoder.xlsx', index=False)

    ######################################################  
    #Cruce Data                                          #         
    ######################################################
    
    # Carga tus datos desde los archivos Excel
    df1 = pd.read_excel('datos_modificados.xlsx')
    data1 = pd.read_excel("resultados_IsolationF.xlsx")
    data2 = pd.read_excel("resultados_LOF.xlsx")
    data3 = pd.read_excel("resultados_autoencoder.xlsx")
    data4 = pd.read_excel("resultados_dbscan.xlsx")



    # Renombrar las columnas de etiqueta antes de fusionar y convertir a minúsculas
    data1.rename(columns={'Etiqueta': 'Etiqueta_IsolationF'}, inplace=True)
    data1['Etiqueta_IsolationF'] = data1['Etiqueta_IsolationF'].str.lower()

    data2.rename(columns={'anomalia': 'Etiqueta_LOF'}, inplace=True)
    data2['Etiqueta_LOF'] = data2['Etiqueta_LOF'].str.lower()

    data3.rename(columns={'Etiqueta': 'Etiqueta_autoencoder'}, inplace=True)
    data3['Etiqueta_autoencoder'] = data3['Etiqueta_autoencoder'].str.lower()

    data4.rename(columns={'Etiqueta': 'Etiqueta_dbscan'}, inplace=True)
    data4['Etiqueta_dbscan'] = data4['Etiqueta_dbscan'].str.lower()

    # dropear columnas que no se van a utilizar

    columns_to_drop = ['UsuarioEntrada', 'Month', 'MonthDay', 'WeekDay','Unnamed: 0'] 
    data2.drop(columns_to_drop, axis=1, inplace=True)
    data3.drop(columns_to_drop, axis=1, inplace=True)
    data4.drop(columns_to_drop, axis=1, inplace=True)  
    # data4.head() 
    # Fusionar los datasets en uno solo basado en las columnas comunes
    etiquetas = ['Etiqueta_IsolationF','Etiqueta_LOF', 'Etiqueta_autoencoder', 'Etiqueta_dbscan']

    """ merged_df = pd.merge(data1, data2, on=['Sucursal', 'HoraEntrada',
                                        'below_5', 'between_5_and_30', 'between_30_and_60', 'between_60_and_90',
                                        'above_90', 'transaction_1', 'transaction_2', 'transaction_3'], how='inner')
    print(merged_df)
    merged_df = pd.merge(merged_df, data3, on=['Sucursal', 'HoraEntrada',
                                            'below_5', 'between_5_and_30', 'between_30_and_60', 'between_60_and_90',
                                            'above_90', 'transaction_1', 'transaction_2', 'transaction_3'], how='inner')

    print(merged_df)
    merged_df = pd.merge(merged_df, data4, on=['Sucursal', 'HoraEntrada',
                                            'below_5', 'between_5_and_30', 'between_30_and_60', 'between_60_and_90',
                                            'above_90', 'transaction_1', 'transaction_2', 'transaction_3'], how='inner')
    print(merged_df) """

    merged_df = pd.concat([data1['Etiqueta_IsolationF'], data2['Etiqueta_LOF'], data3['Etiqueta_autoencoder'], data4['Etiqueta_dbscan']], axis=1)

    merged_df = df1.merge(merged_df, left_index=True, right_index=True)

    # Obtener el número de etiquetas "anormal" para cada fila
    merged_df['Num_Anormal'] = merged_df[etiquetas].apply(lambda x: sum(x.values == 'anormal'), axis=1)

    # Agregar columna "Fraude" según las condiciones dadas
    merged_df['Fraude'] = merged_df['Num_Anormal'].map({4: 'Muy Anormal', 3: 'Muy Anormal', 2: 'Muy Anormal', 1: 'Anormal', 0: 'Normal'})

    # Eliminar columnas innecesarias
    merged_df.drop(columns=['Num_Anormal'], inplace=True)

    # Guardar el resultado del cruce en un archivo Excel
    merged_df.to_excel("resultados_cruce.xlsx", index=False)

# Después de completar el entrenamiento:
st.success('Listo!')