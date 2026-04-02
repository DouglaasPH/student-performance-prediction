import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def process_data(df):
    print("DataFrame info:")
    print(df.head()) # ver primeiras linhas do dataframe
    print(df.info()) # tipos de dados e estrutura
    print(df.describe(include="object")) # estatísticas descritivas para colunas numéricas e categóricas
    
    # MISSING VALUES
    print(df.isnull().sum()) # verificar quantidade de valores nulos por coluna
    
    # ENCODING
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class']) # transformar a coluna 'Class' em valores numéricos -> L, M ou H vira 0, 1 ou 2
    
    # ONE HOT ENCODING
    df = pd.get_dummies(df, drop_first=True) # transformar colunas categóricas em colunas numéricas (0 ou 1) e eliminar a primeira categoria para evitar multicolinearidade
    
    print(df.corr()) # análise de correlação entre variáveis
    
    # HEATMAP
    plt.figure(figsize=(50,40))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm') # mapa de calor para visualizar correlações
    plt.show()
    
    # CORRELATION WITH TARGET
    df.corr()['Class'].sort_values(ascending=False) # correlação das variáveis com a variável alvo 'Class'

    # MULTICOLLINEARITY REMOVAL
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # matriz de correlação superior para identificar pares de variáveis altamente correlacionadas
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)] # identificar colunas para remover com base em correlação maior que 0.9
    df.drop(to_drop, axis=1, inplace=True) # remover colunas altamente correlacionadas para evitar multicolinearidade
    print("Colunas removidas por multicolinearidade:", to_drop)
        
    return df