import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

from data_loader import load_data
from data_science import enconding_categorical, enconding_class_train, process_data, remove_multicollinearity_predict, remove_multicollinearity_train

# ===================================
# CARREGAR DADOS
# ===================================
@st.cache_data
def get_data():
    return load_data()

# ===================================
# TREINAR MODELO
# ===================================
@st.cache_resource
def train_model():
    df = load_data()
    df = process_data(df)

    # Encoding target
    df, le = enconding_class_train(df)

    # One hot encoding
    df = enconding_categorical(df)

    # Remover multicolinearidade
    df, to_drop = remove_multicollinearity_train(df)

    # Preparar dados ML
    X_train, X_test, y_train, y_test, X, y = prepare_data(df)

    # Treinar modelo
    model = hyperparameter_tuning(X_train, y_train)["Random Forest Classifier"]

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Salvar colunas
    train_columns = X.columns

    return model, scaler, train_columns, to_drop, le

# carregar dados e modelo
data = get_data()
model, scaler, train_columns, to_drop, le = train_model()

# ===================================
# TÍTULO
# ===================================
st.title("Student Performance Prediction")
st.markdown("Este sistema utiliza técnicas de Machine Learning para prever o desempenho de estudantes com base em informações acadêmicas, comportamentais e familiares.")

# ===================================
# COMO O MODELO DECIDE
# ===================================
st.header("Como o modelo toma decisões")

st.write("""
O modelo analisa principalmente o engajamento do aluno, participação em aula,
acesso a materiais, frequência escolar e envolvimento dos pais.

Alunos que:
- Participam mais das aulas
- Acessam mais materiais
- Visualizam avisos
- Participam de discussões
- Faltam menos
- Têm maior envolvimento dos pais

tendem a apresentar desempenho **High**.
""")

# ===================================
# FEATURE IMPORTANCE
# ===================================
st.header("Variáveis mais importantes")

st.write("""
A análise de Feature Importance mostrou que as variáveis que mais influenciam
a previsão do modelo são:
""")

st.write("""
1. VisitedResources
2. raisedhands
3. AnnouncementsView
4. StudentAbsenceDays
5. Discussion
6. ParentAnsweringSurvey
7. Relation (Mãe/Pai)
8. ParentschoolSatisfaction
""")

# ===================================
# INTERPRETAÇÃO
# ===================================
st.subheader("Interpretação")

st.write("""
O desempenho do aluno está fortemente relacionado a:

- Engajamento com a plataforma
- Participação em aula
- Participação em discussões
- Frequência escolar
- Envolvimento dos pais
- Satisfação dos pais com a escola

Isso indica que fatores comportamentais têm maior impacto no desempenho
acadêmico do que fatores demográficos.
""")

# ===================================
# MOSTRAR DATASET
# ===================================
st.subheader("Dataset")

cols = st.multiselect(
    "Selecione colunas",
    data.columns.tolist(),
    default=data.columns.tolist()[:5]
)

st.dataframe(data[cols].head(10))

# ===================================
# GRÁFICO
# ===================================
st.subheader("Distribuição das Classes")

fig = px.histogram(data, x="Class")
st.plotly_chart(fig)

# ===================================
# SIDEBAR - INPUTS
# ===================================
st.sidebar.subheader("Dados do Estudante")

raisedhands = st.sidebar.slider("Raised Hands", 0, 100, 10)
visited = st.sidebar.slider("Visited Resources", 0, 100, 10)
announcements = st.sidebar.slider("Announcements View", 0, 100, 10)
discussion = st.sidebar.slider("Discussion", 0, 100, 10)

# novas
nationality = st.sidebar.selectbox("Nationality", ["KW", "lebanon", "Egypt", "SaudiArabia", "USA", "Iran", "Syria", "Jordan", "venzuela", "Iraq"])
birthplace = st.sidebar.selectbox("Place of Birth", [     'KuwaIT',     'lebanon','Egypt', 'SaudiArabia',  'USA', 'Jordan',    'venzuela', 'Iran','Tunis',     'Morocco', 'Syria', 'Iraq',   'Palestine','Lybia'])
stage = st.sidebar.selectbox("StageID", ["lowerlevel", "MiddleSchool", "HighSchool"])
grade = st.sidebar.selectbox("GradeID", ["G-01", "G-02", "G-03", "G-04", "G-05", "G-06", "G-07", "G-08", "G-09", "G-10", "G-11", "G-12"])
section = st.sidebar.selectbox("SectionID", ["A", "B", "C"])
topic = st.sidebar.selectbox("Topic", ['IT','Math', 'Arabic','Science','English', 'Quran', 'Spanish', 'French', 'History', 'Biology', 'Chemistry', 'Geology'])
relation = st.sidebar.selectbox("Relation", ["Father", "Mum"])
parent_survey = st.sidebar.selectbox("Parent Answering Survey", ["Yes", "No"])
parent_satisfaction = st.sidebar.selectbox("Parent School Satisfaction", ["Good", "Bad"])
gender = st.sidebar.selectbox("Gender", ["M", "F"])
semester = st.sidebar.selectbox("Semester", ["F", "S"])
absence = st.sidebar.selectbox("Student Absence Days", ["Under-7", "Above-7"])

btn_predict = st.sidebar.button("Prever desempenho")

# ===================================
# PREDIÇÃO
# ===================================
if btn_predict:

    input_dict = {
        "gender": gender,
        "NationalITy": nationality,
        "PlaceofBirth": birthplace,
        "StageID": stage,
        "GradeID": grade,
        "SectionID": section,
        "Topic": topic,
        "Semester": semester,
        "Relation": relation,
        "ParentAnsweringSurvey": parent_survey,
        "ParentschoolSatisfaction": parent_satisfaction,
        "StudentAbsenceDays": absence,
        "raisedhands": raisedhands,
        "VisITedResources": visited,
        "AnnouncementsView": announcements,
        "Discussion": discussion
    }

    df_input = pd.DataFrame([input_dict])

    # ONE HOT
    df_input = pd.get_dummies(df_input)

    # alinhar colunas com treino
    df_input = df_input.reindex(columns=train_columns, fill_value=0)

    # remover multicolinearidade
    df_input = remove_multicollinearity_predict(df_input, to_drop)

    # scaling
    df_input = scaler.transform(df_input)

    # previsão
    prediction = model.predict(df_input)[0]
    print(prediction)

    mapping_display = {
        "L": "Low",
        "M": "Medium",
        "H": "High"
    }

    prediction_label = le.inverse_transform([prediction])[0]
    st.success(mapping_display[prediction_label])
