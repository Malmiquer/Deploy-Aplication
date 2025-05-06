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

            # **4. Detecção de Anomalias**
    st.header("3. Detecção de Anomalias")

        # Função para criar a coluna com a presença de máquinas com anomalias
def detecta(potencias, min_val, max_val, PFEJ, PFEZ1, PFEp1, PFEp2):
            """
            Detecta anomalias na lista de potências com base nos limites e nas perdas.
            """
            for valor in potencias:
                if valor < min_val or valor > max_val or PFEJ >= 0.03* valor or PFEZ1 >= 0.03* valor or PFEp1 >= 0.03* valor or PFEp2 >= 0.03* valor :
                    return 1  # Anomalia detectada
                return 0  # Sem anomalia
            
        # Função para criar a coluna com a presença de máquinas com anomalias
            # Aplicando a função para criar a coluna 'Flag_anomalia'
            agrupado['Flag_anomalia'] = agrupado.apply(
                    lambda row: detecta(row['Potência'], row['Min_Potencia'], row['Max_Potencia'], row['PFEJ'], row['PFEZ1'],  row['PFEp1'],  row['PFEp2']),
                    axis=1
                    )

# Visualizar os resultados
            st.write("Resultados da Detecção de Anomalias:")
            st.dataframe(agrupado[['Material', 'Potência', 'Min_Potencia', 'Max_Potencia', 'PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2', 'Flag_anomalia']])

            # Modelo de isolamento
            model = IsolationForest(n_estimators=50, contamination=0.1)
            model.fit(agrupado[['PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2']])

            agrupado['scores'] = model.decision_function(agrupado[['PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2']])
            agrupado['anomaly'] = model.predict(agrupado[['PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2']])

            print(agrupado.head(20))

            st.write("Resultados da Detecção de Anomalias:")
            st.dataframe(agrupado[['Material', 'scores', 'anomaly']])

# Definindo a classe MotorInducao
class MotorInducao:
    def __init__(self, Voltage, Current, Frequency, Fator, PFEJ, PFEZ1, PFEp1, PFEp2, d2):
        self.Voltage = Voltage
        self.Current = Current
        self.Frequency = Frequency
        self.Fator = Fator
        self.PFEJ = PFEJ
        self.PFEZ1 = PFEZ1
        self.PFEp1 = PFEp1
        self.PFEp2 = PFEp2
        self.d2 = d2

    def calcular_perdas_ferro(self):
        perdas_totais = self.PFEJ + self.PFEZ1 * self.d2 + self.PFEp1 + self.PFEp2 
        st.subheader("Cálculo de Perdas Totais no Ferro")
        st.write(f"Perdas Totais de Ferro, em Vazio: {perdas_totais:.2f} W")
        return perdas_totais

    def causas_perdas_ferro(self):
        st.subheader("Principais Causas das Perdas no Ferro")
        causas = []
        if self.PFEJ > 0.03 * self.Fator:
            causas.append(f"Perda na coroa estator (PFEJ): {self.PFEJ:.2f} W")
        if self.PFEZ1 > 0.03 * self.Fator:
            causas.append(f"Perda no dente estator fluxo principal (PFEZ1): {self.PFEZ1:.2f} W")
        if self.PFEp1 > 0.03 * self.Fator:
            causas.append(f"Perda no dente estator pulsação (PFEp1): {self.PFEp1:.2f} W")
        if self.PFEp2 > 0.03 * self.Fator:
            causas.append(f"Perda no dente rotor pulsação com amortecimento da gaiola (PFEp2): {self.PFEp2:.2f} W")
        for causa in causas:
            st.write(causa)
        return causas

# Interface do Streamlit
    st.header("4. Análise de Perdas em Motores de Indução")
    st.sidebar.subheader("Parâmetros do Motor")
                # Entrada de dados do usuário
voltage = st.sidebar.number_input("Tensão (V):", min_value=0, step=10)
current = st.sidebar.number_input("Corrente (A):", min_value=0.0, step=0.5)
Potência = st.sidebar.number_input("Potência (W):", min_value=0, step=1)
frequency = st.sidebar.number_input("Frequência (Hz):", min_value=0, step=1)
fator = st.sidebar.number_input("Fator de Correção:", min_value=0.0, step=0.01)
PFEJ = st.sidebar.number_input("PFEJ (W):", min_value=0.0, step=0.5)
PFEZ1 = st.sidebar.number_input("PFEZ1 (W):", min_value=0.0, step=0.5)
PFEp1 = st.sidebar.number_input("PFEp1 (W):", min_value=0.0, step=0.5)
PFEp2 = st.sidebar.number_input("PFEp2 (W):", min_value=0.0, step=0.5)
d2 = st.sidebar.number_input("d2:", min_value=0.0, step=0.5)


# Criar instância do motor com os dados fornecidos
motor = MotorInducao(voltage, current, frequency, fator, PFEp1, PFEp2, PFEJ, PFEZ1, d2)
if uploaded_file is not None:
    
    df['CPFEJ'] = df['FJ'] * (df['KC'] ** 2) * (
            (df['kch'] * df['V10'] * df['a'] * (df['Frequency'] / 50)) +
            (df['kce'] * df['V10'] * (1 - df['a']) * (df['Frequency'] / 50) ** 2)
        )
    df['PFEJ'] = df['CPFEJ'] * (df['BJ1'] ** 2) * df['GJ1']

    df['PFEZ1'] = df['CPFEZ1'] * (df['BZ1'] ** 2) * df['GZ1']

    df['PFEp1'] = df['FZ'] * df['V10'] * ((df['a'] * (df['fp1'] / 50) +
    (1 - df['a']) * (df['fp1'] / 50) ** 2)) * (df['Bp1'] ** 2) * df['GZ1']
    
    df['PFEp2'] = (df['d2'] * df['FZ'] * df['V10']) * ((df['a'] * (df['fp2'] / 50) +
    (1 - df['a']) * (df['fp2'] / 50) ** 2)) * ((df['Bp2'] ** 2) * df['GZ2'])

    # df['CPFEJ'] = df['FJ'] * (df['KC'] ** 2)  
    # df['PFEJ'] = df['CPFEJ'] * (df['BJ1'] ** 2) * df['GJ1']
    # df['PFEZ1'] = df['FZ'] * (df['KC'] ** 2) * ((df['BZ1'] ** 2)) * df['GZ1']
    # df['PFEp1'] = df['FZ'] * df['V10'] * ((df['BZ1'] ** 2)) * df['GZ1']
    # df['PFEp2'] = df['FZ'] * df['V10'] * ((df['BZ2'] ** 2)) * df['GZ2']

    PFEJ = df['PFEJ'].mean()
    PFEZ1 = df['PFEZ1'].mean()
    PFEp1 = df['PFEp1'].mean()
    PFEp2 = df['PFEp2'].mean()
    d2 = df['d2'].mean()

# Botão para calcular e exibir resultados
if st.button("Click"):
    perdas_totais = motor.calcular_perdas_ferro()
    causas = motor.causas_perdas_ferro()


    # **5. Visualizações Gráficas**
    st.header("5. Visualizações Gráficas")

    # Histogramas com Plotly 'PFEJ', 'PFEZ1', 'PFEp1', 'PFEp2'
   
    fig_PFEJ = px.histogram(df, x='PFEJ', nbins=20, title="Distribuição das Perdas na coroa estator (PFEJ)")
    st.plotly_chart(fig_PFEJ, use_container_width=True)

    fig_PFEZ1 = px.histogram(df, x='PFEZ1', nbins=20, title="Distribuição das Perdas no dente estator fluxo principal (PFEZ1)")
    st.plotly_chart(fig_PFEZ1, use_container_width=True)
    
    fig_PFEp1 = px.histogram(df, x='PFEp1', nbins=20, title="Distribuição das Perdas no dente estator pulsacao (PFEp1)")
    st.plotly_chart(fig_PFEp1, use_container_width=True)

    fig_PFEp2 = px.histogram(df, x='PFEp2', nbins=20, title="Distribuição das Perdas no dente rotor pulsação com amortecimento da gaiola (PFEp2)")
    st.plotly_chart(fig_PFEp2, use_container_width=True)

    # Gráfico de dispersão
    fig_disp = px.scatter(df, x='Potência', y='PFEJ', color='Material',
                              title="PFEJ vs Potência")
    st.plotly_chart(fig_disp, use_container_width=True)

    fig_disp = px.scatter(df, x='Potência', y='PFEZ1', color='Material',
                              title="PFEZ1 vs Potência")
    st.plotly_chart(fig_disp, use_container_width=True)

    fig_disp = px.scatter(df, x='Potência', y='PFEp1', color='Material',
                              title="PFEp1 vs Potência")
    st.plotly_chart(fig_disp, use_container_width=True)

    fig_disp = px.scatter(df, x='Potência', y='PFEp2', color='Material',
                              title="PFEp2 vs Potência")
    st.plotly_chart(fig_disp, use_container_width=True)

    # Correlações
    st.subheader("Mapa de Correlações")
    numerical_df = agrupado.select_dtypes(include=['number'])
    fig_corr, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

####   PROPHET   #######
#     # Upload de arquivo
#     st.sidebar.title("Upload de Dados")
#     uploaded_file = st.sidebar.file_uploader("Carregar arquivo", type=["xlsx","csv"])

#     # Verificar se um arquivo foi carregado
# if uploaded_file is not None:
#     #Carregar o arquivo excel
#     df = pd.read_excel(uploaded_file)
#     # Preparar dados para Prophet
#     df_pred = df.rename(columns={'Perda_Ensaio': 'y', 'DATE': 'ds'})
#     df_pred['ds'] = pd.to_datetime(df_pred['ds'], errors='coerce')
#     df_pred = df_pred[['ds', 'y']].dropna()

#         # Treinar modelo
#     modelo = Prophet()

#     modelo.fit(df_pred)

#     # Fazer previsões
#     # Criar dataframe futuro
#     periods = st.sidebar.slider("Número de períodos para previsão", 1, 24, 12)  # Slider interativo
#     future = modelo.make_future_dataframe(periods=periods)
#     previsao = modelo.predict(future)

#     #Exibir gráfico de previsão
#     fig_forecast = modelo.plot(previsao)
#     st.pyplot(fig_forecast)

#     st.subheader("Previsão")
#     st.dataframe(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

#     # Plotar componentes da previsão
#     st.subheader("Componentes da Previsão")
#     st.plotly_chart(plot_components_plotly(modelo, previsao))


    # **7. Download do Arquivo Processado**
    st.header("Baixar Dados Processados")

    st.download_button(
                label="Baixar Dados Processados",
                data=agrupado.to_csv(index=False).encode('utf-8'),
                file_name="dados_processados.csv",
                mime="text/csv"
            )
else:
    st.info("Por favor, carregue um arquivo Excel para começar.")

### METODO KNN ###
    st.header("7. Regressão KNN")
    st.write("Este aplicativo utiliza o algoritmo de Regressão KNN com ajuste de hiperparâmetros.")

    default_test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    seed = st.sidebar.number_input("Random State Seed", min_value=0, value=42)

    # Simulação de dados
    st.sidebar.subheader("Generate Data")
    a = np.random.rand(1000) * 100
    b = 3 * a + np.random.normal(0, 10, 1000)

    # Divisão dos dados
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=default_test_size, random_state=seed)

    # Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor())
    ])

    # Hiperparâmetros
    hyperparameters = {
        'regressor__n_neighbors': [2, 3, 5, 10],
        'regressor__weights': ['uniform', 'distance'],
        'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        pipe,
        param_grid=hyperparameters,
        return_train_score=True,
        scoring='neg_root_mean_squared_error',
        n_jobs=-2,
        cv=5
    )
    # Ajustando modelo
    a_train_reshaped = a_train.reshape(-1, 1)
    grid_search.fit(a_train_reshaped, b_train)

    # Melhores hiperparâmetros
    cv_best_params = grid_search.best_params_
    st.write("Best Hyperparameters")
    st.write(cv_best_params)

    # Configurando o pipeline com melhores parâmetros
    pipe.set_params(**cv_best_params)
    pipe.fit(a_train_reshaped, b_train)

    # Previsões
    b_test_pred = pipe.predict(a_test.reshape(-1, 1))

    # Métricas
    rmse_test = math.sqrt(mean_squared_error(b_test, b_test_pred))
    mae_test = mean_absolute_error(b_test, b_test_pred)
    mape_test = mean_absolute_percentage_error(b_test, b_test_pred)
    r2_test = r2_score(b_test, b_test_pred)

    df_metricas = pd.DataFrame(data={
        'RSME': [rmse_test],
        'MAE': [mae_test],
        'MAPE': [mape_test],
        'R²': [r2_test]
    })
    st.write("Metrics")
    st.table(df_metricas)
    # Comparação de dados reais e previstos
    b_pred = pd.DataFrame(data=pipe.predict(a_test.reshape(-1, 1)), columns=['Predicted Values'])
    b_real = pd.DataFrame(data=b_test, columns=['Real Values'])
    df_comparison = pd.concat([b_real, b_pred], axis=1)
    df_comparison.columns = ['Real_Data', 'Predicted_Value']
    df_comparison['Percentage_difference'] = 100 * (df_comparison['Predicted_Value'] - df_comparison['Real_Data']) / df_comparison['Real_Data']
    df_comparison['Average'] = df_comparison['Real_Data'].mean()
    df_comparison['Q1'] = df_comparison['Real_Data'].quantile(0.25)
    df_comparison['Q3'] = df_comparison['Real_Data'].quantile(0.75)
    df_comparison['USL'] = df_comparison['Real_Data'].mean() + 2 * df_comparison['Real_Data'].std()
    df_comparison['LSL'] = df_comparison['Real_Data'].mean() - 2 * df_comparison['Real_Data'].std()

    st.write("### Real vs Predicted Comparison")
    st.dataframe(df_comparison)

    # Gráficos
    st.write("### Plots")

    # Gráfico 1: Valores reais vs previstos
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.set_title('Real Value vs Predicted Value', fontsize=25)
    ax.plot(df_comparison.index, df_comparison['Real_Data'], label='Real', marker='D', markersize=10, linewidth=0)
    ax.plot(df_comparison.index, df_comparison['Predicted_Value'], label='Predicted', c='r', linewidth=1.5)
    ax.plot(df_comparison.index, df_comparison['Average'], label='Mean', linestyle='dashed', c='yellow')
    ax.plot(df_comparison.index, df_comparison['Q1'], label='Q1', linestyle='dashed', c='g')
    ax.plot(df_comparison.index, df_comparison['Q3'], label='Q3', linestyle='dashed', c='g')
    ax.plot(df_comparison.index, df_comparison['USL'], label='USL', linestyle='dashed', c='r')
    ax.plot(df_comparison.index, df_comparison['LSL'], label='LSL', linestyle='dashed', c='r')
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    st.pyplot(fig)

    # Gráfico 2: Dispersão entre real e previsto
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.set_title('Real Value vs Predicted Value (Scatter)', fontsize=25)
    ax.scatter(df_comparison['Real_Data'], df_comparison['Predicted_Value'], s=100)
    ax.plot(df_comparison['Real_Data'], df_comparison['Real_Data'], c='r')
    ax.set_xlabel('Real', fontsize=25)
    ax.set_ylabel('Predicted', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    st.pyplot(fig) 



# Configuração da aplicação Streamlit
st.markdown(
    """
    <style>
    .header {
        font-size: 32px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
    }
    </style>
    <div class="header">Análise de Perdas em Motores de Indução</div>
    """,
    unsafe_allow_html=True
)

#Divida a Página em Colunas
col1, col2, col3, col4 = st.columns(4)

#Gráfico na primeira coluna
with col1:
    st.subheader("Gráfico de Potência")
    st.line_chart(df["Potência"])

with col2:
    st.subheader("Gráfico das Perda Por Histerese")
    st.bar_chart(df["PFEJ"])

with col3:
    st.subheader("Gráfico das Perdas por Correntes Parasitas")
    st.bar_chart(df["PFEZ1"])

# Tabela na quarta coluna
with col4:
    st.subheader("Resumo dos Dados")
    st.dataframe(df.describe())

#Customize as Cores e Estilos com o st.markdown
tema = st.sidebar.radio("Escolha o tema:", ["Claro", "Escuro"])

if tema == "Claro":
    st.markdown('<style>body { background-color: #FFFFFF; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body { background-color: #2E2E2E; color: white; }</style>', unsafe_allow_html=True)
st.title("Fórmulas Matemáticas Utilizadas")

st.markdown(r"""
## Introdução
Abaixo estão alguns exemplos de fórmulas matemáticas utilizadas no Cálculo:
### Fórmula de KC
$$
KC = KC1.KC2
$$
### Fórmula de TN1
$$
TN1 = (\pi.Di)/N1
$$
### Fórmula de TN2
$$
TN2 = (\pi.Di)/N2
$$
### Fórmula de fatc
$$
fatc = \frac{De - Di}{Di}
$$
### Fórmula de  kch
$$
Kch = e^{0.019.Polos^2.fatc^{1.5}}
$$
### Fórmula de kce
$$
Kce = e^{0.1.Polos^1.8.fatc^{1.4}}
$$
### Fórmula de CPFEJ
$$
CPFEJ = FJ.KC^2.(kch.V10.anhys.(Frequency/50) + kce.V10.(1 - anhys).(Frequency/50)^2)
$$
### Fórmula de CPFEZ1
$$
CPFEZ1 = FZ.KC^2.(V10.anhys.(Frequency/50) + V10.(1 - anhys).(Frequency/50)^2)
$$
### Fórmula de Bp1
$$
Bp1 = \frac{TN2}{2.TN1}.(\frac{Kc2 - 1}{Kc2}).BZ1
$$
### Fórmula de Bp2
$$
Bp2 = \frac{TN1}{2.TN2}.(\frac{Kc1 - 1}{Kc1}).BZ2
$$
### Fórmula de fp1
$$
fp1 = N2.(Frequency/Polos)
$$
### Fórmula de fp2
$$
fp2 = N1.(Frequency/Polos)
$$
### Fórmula de PFEJ
$$
PFEJ = CPFEJ.BJ1^2.GJ1
$$
### Fórmula de PFEZ1
$$
PFEZ1 = CPFEZ1.BZ1^2.GZ1
$$
### Fórmula de PFEp1
$$
PFEp1 = FZ.V10.(a.(fp1/50) + (1 - a).(fp1/50)^2).Bp1^2.GZ1
$$
### Fórmula de PFEp2
$$
PFEp2 = d2.FZ.V10.(a.(fp2/50) + (1 - a).(fp2/50)^2).Bp2^2.GZ2
$$
### Fórmula da Perda Total em Vazio
$$
Perda_Total = PFEJ + PFEZ1 + PFEp1 + PFEp2.d2
$$
""")

#Adicione Expansores para Informações Adicionais
with st.expander("Clique aqui para mais informações"):
    st.write("Você acabou de calcular as Perdas no Ferro!")
