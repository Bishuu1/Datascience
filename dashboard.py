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


# funciones xavi
def generate_user_dict(operators_df):
    users_id = operators_df[0]
    users_name = operators_df[1]
    users_dict = {}
    for index in range(len(users_name)):
        if users_name[index] in users_dict:
            continue
        users_dict[users_name[index].lower()] = users_id[index]
    return users_dict


def generate_branch_dict(branches_df):
    branch_id = branches_df[0]
    branch_name = branches_df[1]
    branch_dict = {}
    for index in range(len(branch_name)):
        if branch_name[index] in branch_dict:
            continue
        branch_dict[branch_name[index].lower()] = branch_id[index]
    return branch_dict


def switch_name_for_id(user_list: list, name_dict: dict):
    user_id_list = []
    for user in user_list:
        if type(user) != str:
            continue
        user_id_list.append(name_dict[user.lower()])
    return user_id_list


def count_transactions_by_hour(dataframe):
    transaction_hour_count = dataframe.groupby(["Sucursal", "UsuarioEntrada", "HoraEntrada"], as_index=False).agg(
        below_5=("Minutos", lambda x: (x <= 5).sum()),
        between_5_and_30=("Minutos", lambda x: (x <= 30).sum()),
        between_30_and_60=("Minutos", lambda x: (x <= 60).sum()),
        between_60_and_90=("Minutos", lambda x: (x <= 90).sum()),
        above_90=("Minutos", lambda x: (x > 90).sum()),
    )
    transaction_hour_count["between_60_and_90"] -= transaction_hour_count[
        "between_30_and_60"
    ]
    transaction_hour_count["between_30_and_60"] -= transaction_hour_count[
        "between_5_and_30"
    ]
    transaction_hour_count["between_5_and_30"] -= transaction_hour_count["below_5"]
    return transaction_hour_count


def count_operations_by_hour(dataframe):
    transaction_hour_count = dataframe.groupby(["Fecha", "Hora", "branch_id", "user_id"], as_index=False).agg(
        transaction_1=("session_type", lambda x: (x == 1).sum()),
        transaction_2=("session_type", lambda x: (x == 2).sum()),
        transaction_3=("session_type", lambda x: (x == 3).sum()),
    )
    return transaction_hour_count


# Configuración de la página
st.title('Detección de fraudes en estacionamientos')

# Importing the data
trans_file = st.file_uploader("subir transacciones", type=['csv', 'xlsx'])
operator_file = st.file_uploader("subir operadores", type=['csv', 'xlsx'])
branches_file = st.file_uploader("subir sucursales", type=['csv', 'xlsx'])
sessions_file = st.file_uploader("subir sesiones", type=['csv', 'xlsx'])

if trans_file is not None and operator_file is not None and branches_file is not None and sessions_file is not None:

    transactions_df = pd.read_excel(trans_file) if trans_file.name.endswith('.xlsx') else pd.read_csv(trans_file)
    operators_df = pd.read_csv(operator_file, sep=';', header=None) if operator_file.name.endswith(
        '.csv') else pd.read_excel(operator_file)
    branches_df = pd.read_csv(branches_file, sep=';', header=None) if branches_file.name.endswith(
        '.csv') else pd.read_excel(branches_file)
    sessions_df = pd.read_csv(sessions_file, sep=';', header=0) if sessions_file.name.endswith(
        '.csv') else pd.read_excel(sessions_file)
    if st.button('Procesar'):

        users_dict = generate_user_dict(operators_df)
        usuarios_entrada = list(transactions_df.UsuarioEntrada)
        usuarios_salida = list(transactions_df.UsuarioSalida)
        branches_dict = generate_branch_dict(branches_df)
        sucursales = list(transactions_df.Sucursal)

        usuarios_entrada = switch_name_for_id(usuarios_entrada, users_dict)
        usuarios_salida = switch_name_for_id(usuarios_salida, users_dict)
        # sucursales =
        # Modificar el dato de la columna edad de cada fila
        columns_to_drop = [
            "Calza",
            "TicketLiberado",
            "DueñodeTarjeta",
            "LPREntrada",
            "LPRSalida",
            "IDReserva",
            "NombreDescuento",
            "CuponDescuento",
            "URL",
        ]

        transactions_df = transactions_df.drop(columns_to_drop, axis=1)
        new_transactions_df = pd.DataFrame()

        descuentos = list(transactions_df.Descuento)
        minutos = list(transactions_df.Minutos)
        hora_entrada = set(transactions_df.HoraEntrada)
        hora_salida = set(transactions_df.HoraSalida)
        patentes = list(transactions_df.Patente)

        new_df = pd.DataFrame()
        sucursales = set(transactions_df.Sucursal)
        descuentos = set(transactions_df.Descuento)
        minutos = set(transactions_df.Minutos)

        transactions_df.sort_values(by=["FechaEntrada", "HoraEntrada"])
        transactions_df["HoraEntrada"] = transactions_df["HoraEntrada"].map(
            lambda x: x.split(":")[0]
        )
        transactions_df["FechaEntrada"] = transactions_df["FechaEntrada"].map(
            lambda x: datetime.datetime.strptime(x, "%d-%m-%Y")
        )
        transactions_df["WeekDay"] = transactions_df["FechaEntrada"].apply(
            lambda x: x.weekday()
        )

        days = transactions_df["FechaEntrada"].unique().tolist()
        for day in days:
            temp_df = transactions_df.loc[transactions_df["FechaEntrada"] == day]

            temp_df = count_transactions_by_hour(temp_df)
            temp_df["Month"] = day.month
            temp_df["MonthDay"] = day.day
            temp_df["WeekDay"] = day.weekday()
            new_df = pd.concat([new_df, temp_df], axis=0)

        for index, row in new_df.iterrows():

            current_sucursal = row.at["Sucursal"].lower()
            current_user = row.at["UsuarioEntrada"].lower()
            new_df.at[index, "Sucursal"] = branches_dict[current_sucursal]
            new_df.at[index, "UsuarioEntrada"] = users_dict[current_user]

        new_sessions_df = pd.DataFrame()
        new_sessions_df["Fecha"] = sessions_df[" created_at"].map(
            lambda x: x.split('T')[0]
        )
        new_sessions_df["Hora"] = sessions_df[" created_at"].map(
            lambda x: x.split('T')[1].split(":")[0]
        )
        new_sessions_df["time"] = sessions_df[" created_at"].map(
            lambda x: x.split('T')[1].split(".")[0]
        )
        new_sessions_df["user_id"] = sessions_df[" user_id"]
        new_sessions_df["branch_id"] = sessions_df["branch_id"]
        new_sessions_df["session_type"] = sessions_df["session_type"]

        second_new_df = pd.DataFrame()
        second_new_df = count_operations_by_hour(new_sessions_df)

        second_new_df["Fecha"] = second_new_df["Fecha"].map(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
        )
        third_new_df = pd.DataFrame()
        days = second_new_df["Fecha"].unique().tolist()
        for day in days:
            temp_df = second_new_df.loc[second_new_df["Fecha"] == day]

            temp_df["Mes"] = day.month
            temp_df["DiaMes"] = day.day
            temp_df["DiaSemana"] = day.weekday()
            third_new_df = pd.concat([third_new_df, temp_df], axis=0)

        writer = pd.merge(
            new_df, third_new_df, how='left',
            left_on=['Sucursal', 'UsuarioEntrada', 'HoraEntrada', 'Month', 'MonthDay', 'WeekDay'],
            right_on=['branch_id', 'user_id', 'Hora', 'Mes', 'DiaMes', 'DiaSemana'])

        columns_to_drop = [
            'Mes',
            'DiaMes',
            'DiaSemana',
            'Hora',
            'branch_id',
            'user_id',
            'Fecha'
        ]
        writer.transaction_1.fillna(value=0, inplace=True)
        writer.transaction_2.fillna(value=0, inplace=True)
        writer.transaction_3.fillna(value=0, inplace=True)

        writer = writer.drop(columns_to_drop, axis=1)

        # Guardar el archivo excel modificado
        pd.DataFrame(writer).to_excel("datos_modificados.xlsx", merge_cells=False)

        # Mostrar el DataFrame en Streamlit
        st.dataframe(writer)

        if st.button('Next'):
            switch_page("1-Isolation")
