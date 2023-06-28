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
st.header('RandomForest')


with st.spinner("Corriendo RandomForest"):
    time.sleep(0.1)
    # Load your dataset into a pandas DataFrame
    df = pd.read_excel('resultados_cruce.xlsx')

    # Split the data into features (X) and labels (y)
    columnas_a_dropear = ['Etiqueta_IsolationF',
                          'Etiqueta_LOF', 'Etiqueta_autoencoder', 'Etiqueta_dbscan']
    df = df.drop(columnas_a_dropear, axis=1)
    X = df.drop('Fraude', axis=1)
    y = df['Fraude']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Generate a classification report
    print(classification_report(y_test, y_pred))

# Después de completar el entrenamiento:
st.success('Listo!')
