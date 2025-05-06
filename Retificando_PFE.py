### Importações ###
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px
from prophet.plot import plot_components_plotly
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from prophet import Prophet
import os 
### Configuração inicial do aplicativo ###
st.set_page_config(
    page_title="Predição de Perdas no Ferro",
    layout="wide"
)
# Use Imagens e Ícones
#st.sidebar.image(r"c:\Users\e-malmiquer\Documents\Material_Malmi\TestesrReais\PFE\MOTOR.png", use_container_width=True)

# Caminho dinâmico da imagem
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(CURRENT_DIR, "MOTOR.png")

# Título e introdução
st.title('Análise de Predição das Perdas No Ferro em Motores de Indução')
st.write("""Este aplicativo permite carregar dados, calcular perdas de histerese e correntes parasitas, detectar anomalias e identificar causas das perdas.""")

### Upload do arquivo ###
uploaded_file = st.sidebar.file_uploader("Adicionar Arquivo Aqui", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Carregar dados
    df = pd.read_excel(uploaded_file)
    
    # Visualização dos Dados
    st.header("1. Visualização dos Dados")
    st.write("Aqui estão os dados carregados do arquivo:")
    st.dataframe(df)

    # Verificar colunas necessárias
    colunas_necessarias = ['Kc1', 'Kc2', 'Di', 'N1', 'N2', 'FJ', 'FZ', 'V10', 'a', 'BJ1', 'BZ1', 'BZ2', 'd2', 'GZ2', 'Polos', 'Frequency', 'KS', 'Material', 'Potência']
    if not all(col in df.columns for col in colunas_necessarias):
        st.error(f"Erro: As colunas necessárias não estão no arquivo. Necessárias: {colunas_necessarias}")
    else:
        # Realizar cálculos
        df['KC'] = df['Kc1'] * df['Kc2']

        # Cálculo do Passo da Ranhura
        df['TN1'] = (df['Di'] * np.pi) / df['N1']
        df['TN2'] = (df['Di'] * np.pi) / df['N2']

        # Cálculo do Fator de Curvatura
        df['fatc'] = (df['De'] - df['Di']) / df['Di']

        # Fatores de Correção (Coroa e Dentes)
        df['kch'] = np.exp(0.019 * (df['Polos'] ** 2) * (df['fatc'] ** 1.5))
        df['kce'] = np.exp(0.1 * (df['Polos'] ** 1.8) * (df['fatc'] ** 1.4))

        # Coeficientes de Perdas
        df['CPFEJ'] = df['FJ'] * (df['KC'] ** 2) * (
            (df['kch'] * df['V10'] * df['a'] * (df['Frequency'] / 50)) +
            (df['kce'] * df['V10'] * (1 - df['a']) * (df['Frequency'] / 50) ** 2)
        )

        df['CPFEZ1'] = df['FZ'] * (df['KC'] ** 2) * (
            df['V10'] * df['a'] * (df['Frequency'] / 50) +
            (df['V10'] * (1 - df['a']) * (df['Frequency'] / 50) ** 2)
        )

        # Pulsação nos dentes devido ao fluxo fundamental
        df['Bp1'] = (df['TN2'] / (2 * df['TN1'])) * ((df['Kc2'] - 1) / df['Kc2']) * df['BZ1']

        df['Bp2'] = (df['TN1'] / ((2 * df['TN2']))) * (((df['Kc1'] - 1) / df['Kc1']))* df['BZ2']

        # Frequências de Pulsação
        df['fp1'] = df['N2'] * (df['Frequency'] / df['Polos'])
        df['fp2'] = df['N1'] * (df['Frequency'] / df['Polos'])

        # Perdas por Histerese e Correntes Parasitas
        df['PFEJ'] = df['CPFEJ'] * (df['BJ1'] ** 2) * df['GJ1']

        df['PFEZ1'] = df['CPFEZ1'] * (df['BZ1'] ** 2) * df['GZ1']

        # Perdas por Pulsação
        df['PFEp1'] = df['FZ'] * df['V10'] * ((df['a'] * (df['fp1'] / 50) +
        (1 - df['a']) * (df['fp1'] / 50) ** 2)) * (df['Bp1'] ** 2) * df['GZ1']


        df['PFEp2'] = (df['d2'] * df['FZ'] * df['V10']) * ((df['a'] * (df['fp2'] / 50) +
        (1 - df['a']) * (df['fp2'] / 50) ** 2)) * ((df['Bp2'] ** 2) * df['GZ2'])

        df['Perda_Total_Predito'] = df['PFEJ'] + df['PFEZ1'] + df['PFEp1'] + (df['PFEp2'] * df ['d2'])
        df['Perda_Total_Ensaio'] = df['Pfe com fator'] + df['Pfe sem fator'] + df['Perda_Ensaio'] + df['Perda_Newton'] + df['Perda Newton com fator']
        df['Desvio_Padrão1'] = df['Perda_Total_Predito'] / df['Perda_Total_Ensaio']
        df['Desvio_Padrão2'] = df['Perda_Total_Ensaio'] / df['Perda_Total_Predito']

        # Exibir cálculos
        st.write("Resultados dos cálculos:")
        st.dataframe(df[['KC', 'TN1', 'TN2', 'fatc', 'kch', 'kce', 'CPFEJ', 'CPFEZ1', 'Bp1', 'Bp2', 'fp1', 'fp2', 'PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2', '%Un','FJ', 'FZ', 'Perda_Total_Predito', 'Perda_Total_Ensaio', 'Desvio_Padrão1', 'Desvio_Padrão2']])


        st.subheader("2.1. Análise Agrupada por Material")
        agrupado = df.groupby('Material').agg({
                    'Potência': list,
                    'PFEJ': 'mean',
                    'PFEZ1': 'mean',
                    'PFEp1': 'mean',
                    'PFEp2': 'mean'
                    }).reset_index()

        agrupado['Min_Potencia'] = agrupado['Potência'].apply(min)
        agrupado['Max_Potencia'] = agrupado['Potência'].apply(max)

        st.write("Dados Agrupados:")
        st.dataframe(agrupado)

    
