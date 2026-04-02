import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def prepare_data(df):
    # FEATURE IMPORTANCE
    corr_target = df.corr()['Class'].abs().sort_values(ascending=False)
    selected_features = corr_target[corr_target > 0.1].index
    print(selected_features)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y) # treinar modelo
    
    importances = pd.Series(model.feature_importances_, index=X.columns) # calcular importância das features
    importances = importances.sort_values(ascending=False) # Ordenar da mais importante para a menos importante
    print(importances)
    
    # FEATURE SELECTION
    selected_features = importances[importances > 0.01].index # selecionar features com importância maior que 0.01
    X = X[selected_features] # Criar novo dataframe apenas com essas features

    print("Features selecionadas:")
    print(selected_features)
    
    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Tamanho treino:", X_train.shape)
    print("Tamanho teste:", X_test.shape)
    
    # SCALING
    scaler = StandardScaler()

    # Ajustar o scaler nos dados de treino e transformar
    X_train = scaler.fit_transform(X_train)

    # Transformar os dados de teste usando o mesmo scaler
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X, y


def train_model(X_train, y_train, models):
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} treinado com sucesso")
        
    return models


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    results = pd.Series(results).sort_values(ascending=False) # Converter para Series e ordenar

    print("\nRanking dos modelos (modelo + accuracy):")
    print(results)


def hyperparameter_tuning(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    param_grid_knn = {
        'n_neighbors': [3,5,7,9,11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    param_grid_dt = {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    param_grid_lr = {
        "C": [0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2']
    }
    param_grid_nb = {
        'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06]
    }
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_svm = GridSearchCV(
        SVC(),
        param_grid_svm,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search_knn = GridSearchCV(
        KNeighborsClassifier(),
        param_grid_knn,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_mlp = GridSearchCV(
        MLPClassifier(max_iter=1000),
        param_grid_mlp,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_dt = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid_dt,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_lr = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid_lr,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_nb = GridSearchCV(
        GaussianNB(),
        param_grid_nb,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_ada = GridSearchCV(
        AdaBoostClassifier(),
        param_grid_ada,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )


    grid_search_rf.fit(X_train, y_train)
    grid_search_svm.fit(X_train, y_train)
    grid_search_knn.fit(X_train, y_train)
    grid_search_mlp.fit(X_train, y_train)
    grid_search_dt.fit(X_train, y_train)
    grid_search_lr.fit(X_train, y_train)
    grid_search_nb.fit(X_train, y_train)
    grid_search_ada.fit(X_train, y_train)


    best_models = {
        "Random Forest Classifier": grid_search_rf.best_estimator_,
        "Support Vector Machine": grid_search_svm.best_estimator_,
        "K-Nearest Neighbors": grid_search_knn.best_estimator_,
        "Multi-Layer Perceptron": grid_search_mlp.best_estimator_,
        "Decision Tree": grid_search_dt.best_estimator_,
        "Logistic Regression": grid_search_lr.best_estimator_,
        "Naive Bayes": grid_search_nb.best_estimator_,
        "AdaBoost": grid_search_ada.best_estimator_
    }
    
    print("Melhores parâmetros Random Forest:", grid_search_rf.best_params_)
    print("Melhores parâmetros SVM:", grid_search_svm.best_params_)
    print("Melhores parâmetros KNN:", grid_search_knn.best_params_)
    print("Melhores parâmetros MLP:", grid_search_mlp.best_params_)
    print("Melhores parâmetros Decision Tree:", grid_search_dt.best_params_)
    print("Melhores parâmetros Logistic Regression:", grid_search_lr.best_params_)
    print("Melhores parâmetros Naive Bayes:", grid_search_nb.best_params_)
    print("Melhores parâmetros AdaBoost:", grid_search_ada.best_params_)
    
    return best_models
