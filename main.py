from model_evaluation import calculate_overall_auc, cross_validation, generate_classification_report, generate_confusion_matrix, generate_final_importance_chart, generate_roc_curve
from data_science import process_data
from data_loader import load_data
from machine_learning import evaluate_models, hyperparameter_tuning, prepare_data, train_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


print("Iniciando processo de download do dataset...")
df = process_data(load_data())
print("Dataset carregado e processado com sucesso.")

print("Iniciando preparação dos dados para machine learning...")
X_train, X_test, y_train, y_test, X, y = prepare_data(df)
print("Preparação dos dados concluída.")

print("Iniciando treinamento dos modelos....")
initial_models = {
    "DT (Decision Tree)": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "LR (Logistic Regression)": LogisticRegression(max_iter=1000),
    "NB (Naive Bayes)": GaussianNB(),
    "SVM": SVC(),
    "MLP (Neural Network)": MLPClassifier(max_iter=1000),
    "RF (Random Forest)": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}


trained_models = train_model(X_train, y_train, initial_models)
print("Treinamento dos modelos concluído.")


print("Iniciando avaliação dos modelos antes do hyperparameter tuning....")
evaluate_models(trained_models, X_test, y_test)
print("Avaliação dos modelos concluída.")


print("Iniciando hyperparameter tuning e treinamento dos modelos...")
best_models = hyperparameter_tuning(X_train, y_train)
print("Treinamento dos modelos concluído.")

print("Iniciando avaliação dos modelos depois do hyperparameter tuning....")
evaluate_models(best_models, X_test, y_test)
print("Avaliação dos modelos concluída.")

print("O modelo final escolhido é o Random Forest Classifier, que apresentou a melhor performance após o hyperparameter tuning.")
final_model = best_models["Random Forest Classifier"]


print("Gerando matriz de confusão para o modelo final escolhido...")
generate_confusion_matrix(final_model, X_test, y_test)
print("Matriz de confusão gerada com sucesso.")

print("Gerando relatório de classificação para o modelo final escolhido...")
generate_classification_report(y_test, final_model.predict(X_test))
print("Relatório de classificação gerado com sucesso.")

print("Gerando curva ROC para o modelo final escolhido...")
y_test_bin, y_score = generate_roc_curve(final_model, X_test, y_test)
print("Curva ROC gerada com sucesso.")

print("Calculando AUC geral para o modelo final escolhido...")
calculate_overall_auc(y_test_bin, y_score)
print("AUC geral calculada com sucesso.")

print("Gerando gráfico de importância das features para o modelo final escolhido...")
generate_final_importance_chart(final_model, X)
print("Gráfico de importância das features gerado com sucesso.")

print("Realizando cross-validation para o modelo final escolhido...")
cross_validation(final_model, X, y)
print("Cross-validation realizada com sucesso.")
