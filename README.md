# Student Performance Prediction

Projeto de Machine Learning para prever o desempenho de estudantes com base em dados educacionais.

---

## Objetivo

Desenvolver um pipeline completo de Machine Learning que inclui:

- Carregamento de dados
- Análise exploratória
- Pré-processamento
- Treinamento de modelos
- Avaliação de desempenho
- Comparação entre diferentes algoritmos

---

## Dataset

O projeto utiliza o dataset:

- **xAPI-Edu-Data**
- Contém informações sobre comportamento e desempenho de estudantes
- Variável alvo: classificação do desempenho (`Class`)

---

## Tecnologias utilizadas

- Python -> 3.12.10
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- KaggleHub

---

## Estrutura do projeto

```text
student-performance-prediction/
│
├── app.py                  # Configurações do streamlit
├── data_loader.py          # Carregamento do dataset
├── data_science.py         # Análise exploratória e pré-processamento
├── machine_learning.py     # Treinamento dos modelos
├── model_evaluation.py     # Métricas e avaliação dos modelos
├── main.py                 # Pipeline principal
├── requirements.txt        # Dependências do projeto
├── .gitignore
└── README.md
```

---

## Pipeline do Projeto

O projeto segue uma pipeline completa de Machine Learning, cobrindo desde a exploração dos dados até a avaliação final do modelo:

### 1. Análise Exploratória

1. **Head** – Visualização inicial do dataset
2. **Info** – Estrutura e tipos de dados
3. **Describe** – Estatísticas descritivas
4. **Missing Values** – Identificação e tratamento de valores ausentes

---

### 2. Pré-processamento

5. **Encoding** – Conversão de variáveis categóricas
6. **Correlation Analysis** – Análise de correlação entre variáveis
7. **Heatmap** – Visualização da matriz de correlação
8. **Correlation with Target** – Relação das features com a variável alvo
9. **Multicollinearity Removal** – Remoção de variáveis altamente correlacionadas
10. **Feature Importance** – Avaliação da importância das variáveis
11. **Feature Selection** – Seleção das features mais relevantes

---

### 3. Preparação dos Dados

12. **Train-Test Split** – Separação dos dados em treino e teste
13. **Scaling** – Normalização das variáveis

---

### 4. Modelagem

14. **Model Training** – Treinamento de múltiplos modelos de Machine Learning
15. **Hyperparameter Tuning** – Otimização de parâmetros utilizando GridSearchCV

---

### 5. Avaliação dos Modelos

17. **Model Evaluation** – Métricas gerais de desempenho
18. **Model Comparison** – Comparação entre diferentes algoritmos
19. **Final Model Selection** – Escolha do melhor modelo

---

### 6. Avaliação Final

20. **Confusion Matrix** – Análise de erros de classificação
21. **Classification Report** – Precision, Recall e F1-score
22. **ROC Curve e AUC** – Capacidade de separação do modelo
23. **Feature Importance** – Interpretação das variáveis mais relevantes
24. **Repeated Stratified K-Fold Cross Validation** - Avaliação robusta do modelo com múltiplas repetições mantendo a proporção das classes

---

# 🧠 Observação

- O projeto utiliza **GridSearchCV** para tuning de hiperparâmetros, garantindo uma busca sistemática pelos melhores parâmetros.
- A validação cruzada é aplicada para reduzir overfitting e melhorar a generalização do modelo.

---

## Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/DouglaasPH/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar apenas o machine learning

```bash
python main.py
```

### 4. Executar machine learning + streamlit

```bash
streamlit run app.py
```

---

## 📈 Resultados

O projeto permite:

- Comparar diferentes algoritmos de ML
- Identificar o modelo com melhor desempenho
- Visualizar métricas de avaliação
- Analisar comportamento dos dados educacionais
