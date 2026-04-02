import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

def generate_confusion_matrix(final_model, X_test, y_test):
    y_pred = final_model.predict(X_test) # Prever as classes para os dados de teste usando o modelo final escolhido
    cm = confusion_matrix(y_test, y_pred) # Gerar a matriz de confusão comparando as classes reais (y_test) com as classes previstas (y_pred) para avaliar o desempenho do modelo final escolhido
    plt.figure(figsize=(6,4)) # Configurar o tamanho da figura para a matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Criar um mapa de calor da matriz de confusão usando a biblioteca seaborn, onde 'annot=True' exibe os valores numéricos dentro das células, 'fmt='d'' formata os números como inteiros e 'cmap='Blues'' define a paleta de cores para o mapa de calor
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()


def generate_classification_report(y_test, y_pred):
    target_names = ['Low - 0', 'Medium - 1', 'High - 2']
    print(classification_report(y_test, y_pred, target_names=target_names))


def generate_roc_curve(final_model, X_test, y_test):
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = final_model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()

    for i in range(3):
        plt.plot(fpr[i], tpr[i], label='Classe %d (AUC = %0.2f)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multiclass')
    plt.legend()
    plt.show()
    
    return y_test_bin, y_score


def calculate_overall_auc(y_test_bin, y_score):
    auc_score = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
    print("AUC geral:", auc_score)


def generate_final_importance_chart(final_model, X):
    # Obter importância das features
    importances = final_model.feature_importances_

    # Criar DataFrame
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })

    # Ordenar da maior para menor
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Mostrar tabela
    print(feature_importance)
    
    plt.figure(figsize=(10,6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.gca().invert_yaxis()
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    

def cross_validation(final_model, X, y):
    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=5,
        random_state=42
    )
    scores = cross_val_score(final_model, X, y, cv=cv, scoring='accuracy')
    print("Accuracy média:", scores.mean())
    print("Desvio padrão:", scores.std())
