import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# só no treino
def process_data(df):
    print("DataFrame info:")
    print(df.head()) # ver primeiras linhas do dataframe
    print(df.info()) # tipos de dados e estrutura
    print(df.describe(include="object")) # estatísticas descritivas para colunas numéricas
    print(df.isnull().sum()) # verificar quantidade de valores nulos por coluna
    
    return df


# treino
def enconding_class_train(df):
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class']) # transformar a coluna 'Class' em valores numéricos
    return df, le


# só treino
def enconding_categorical(df):
    df = pd.get_dummies(df, drop_first=True) # transformar colunas categóricas em colunas numéricas (0 ou 1) e eliminar a primeira categoria para evitar multicolinearidade
    return df


# treino
def eda(df):
    print(df.corr()) # análise de correlação entre variáveis
    
    # HEATMAP
    plt.figure(figsize=(50,40))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm') # mapa de calor para visualizar correlações
    plt.show()
    
    # CORRELATION WITH TARGET
    df.corr()['Class'].sort_values(ascending=False) # correlação das variáveis com a variável alvo 'Class'
    
    return df


# remover
def remove_multicollinearity_train(df, threshold=0.9):
    corr_matrix = df.corr().abs() # matriz de correlação absoluta
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # matriz superior para evitar duplicação
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)] # identificar colunas a serem removidas com base no limiar de correlação
    df.drop(to_drop, axis=1, inplace=True) # remover colunas do dataframe

    print("Colunas removidas por multicolinearidade:", to_drop)

    return df, to_drop


# predição
def remove_multicollinearity_predict(df, to_drop):
    df = df.drop(columns=to_drop, errors='ignore')
    return df
