import xgboost
import mlflow
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nbformat
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,  recall_score, f1_score, precision_score
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import make_classification
from scipy.stats import chi2_contingency
import optuna
import plotly
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform




# Define a function for loading data
def load_data():

# Modify this path based on your actual data file location

 url = "https://raw.githubusercontent.com/Nourben7/machine_learning/main/df_satisfaction.csv"
 data = pd.read_csv(url)
 
 return data

data = load_data()
label_encoder = LabelEncoder()
data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])

#from sklearn.preprocessing import LabelEncoder
numerical_cols = data.select_dtypes(include=['number']).columns
df_numerical = data[numerical_cols] 
                                                                                                                                                                                                                              
##########################################################################################################################
##########################################################################################################################
# --- Modele 1 : KNN Classifier

    

data= data.dropna()
label_encoder = LabelEncoder()
data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
# Data preparation 
# Assuming the last column is the target for demonstration; adjust as needed
X = data.drop(columns=['satisfaction'])
X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
y = data['satisfaction']   # Cible

####################################################
print("Baseline : KNN Classifier")

X, y = X_numeric, y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Pipeline setup with KNeighborsClassifier
pipeline = Pipeline(steps=[('model', KNeighborsClassifier())])

pipeline.fit(X_train, y_train)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Écrire les métriques dans un fichier out/score.txt
if not os.path.exists('out'):  # Vérifier si le répertoire 'out' existe
    os.makedirs('out')

with open('out/score.txt', 'a') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
        
    
accuracy = accuracy_score(y_test, y_pred) * 100  # Pourcentage d'accuracy
print(f'Accuracy (Exactitude): {accuracy:.2f}%')
# Générer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
# Visualiser la matrice de confusion avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel('Prédiction')
plt.ylabel('Vrai Label')
plt.title('Matrice de Confusion du KNN - Baseline')
plt.show()
         

####################################################       
print("Iterations")
    

data = load_data()
data= data.dropna()
label_encoder = LabelEncoder()
data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
print("Normalisation de la donnée")

X = data.drop(columns=['satisfaction']) #'Unnamed: 0','id' 
X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
y = data['satisfaction'] 
X, y = X_numeric, y

# Normalisation de la donnée
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

knn = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.2, random_state=42)


# Entraînement du modèle avec le pipeline
knn.fit(X_train, y_train)
# Prédiction sur l'ensemble de test
y_pred = knn.predict(X_test)

# Accuracy score (test set)
accuracy_test = accuracy_score(y_test, y_pred)

if not os.path.exists('out'):  # Vérifier si le répertoire 'out' existe
    os.makedirs('out')
    
with open('out/score.txt', 'a') as f:
    f.write(f"Accuracy on test set: {accuracy_test:.4f}\n")

print(f'Accuracy on test set: {accuracy_test:.4f}')

# Affichage du rapport de classification (précision, rappel, F1-score)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix on the test set
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set) - Normalisation')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
    
    

####################################################
print("Changement 2: Validation des données")
    # Définir le modèle KNN avec k=10 
k_default = 10
knn = KNeighborsClassifier(n_neighbors=k_default)

# Effectuer la validation croisée
scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')  # 5 plis

    # Écrire les résultats dans un fichier out/score.txt
if not os.path.exists('out'):  # Vérifier si le répertoire 'out' existe
    os.makedirs('out')

with open('out/score.txt', 'a') as f:
    f.write(f"Validation croisée pour k={k_default}:\n")
    f.write(f"Scores: {scores}\n")
    f.write(f"Moyenne de l'accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")

# Afficher les scores pour chaque pli
print(f'Scores de validation croisée pour k={k_default}: {scores}')
print(f'Moyenne de l\'accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')


####################################################
print("Changement 3 - Itération des valeurs de k (n_neighbors)")

X, y = X_numeric, y

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=42)

# Liste pour stocker les scores de précision pour différentes valeurs de k
k_values = list(range(1, 21))  # Tester les valeurs de k de 1 à 20
accuracy_scores = []

# Itérer sur les valeurs de k
for k in k_values:
    # Créer le classificateur KNN avec k voisins
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)  # Entraîner le modèle

    # Générer des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Calculer le score de précision sur l'ensemble de test
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

     # Enregistrer les métriques dans MLflow pour chaque valeur de k
    print(f'Accuracy for k={k}: {accuracy:.4f}')

# Tracer les scores de précision pour différentes valeurs de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid()
plt.show()

# Afficher la meilleure valeur de k basée sur la précision
best_k = k_values[np.argmax(accuracy_scores)]
print(f'Best k value: {best_k} with accuracy: {max(accuracy_scores):.4f}')

# Évaluer le modèle avec le meilleur k sur l'ensemble de test
best_clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)
best_clf.fit(X_train, y_train)
y_test_pred = best_clf.predict(X_test)

# Afficher le rapport de classification sur l'ensemble de test
print("Classification Report for Test Set:")
print(classification_report(y_test, y_test_pred))

# Afficher la matrice de confusion sur l'ensemble de test
cm_train = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion (Ensemble de test) - Iteration de k')
plt.xlabel('Label Prédit')
plt.ylabel('Label Réel')
plt.show()

 
 # Prédictions de probabilités pour l'ensemble de test
y_proba = best_clf.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive (1)

# Calculer les valeurs de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculer l'AUC
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # ligne de hasard
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

     # Écrire les résultats dans un fichier out/score.txt
if not os.path.exists('out'):  # Vérifier si le répertoire 'out' existe
    os.makedirs('out')
    
with open('out/score.txt', 'a') as f:
    f.write(f"Best k value: {best_k} with accuracy: {max(accuracy_scores):.4f}\n")
    f.write(f"Classification Report for Test Set:\n{classification_report(y_test, y_test_pred)}\n")
    
# Sauvegarder les courbes ROC dans le répertoire de sortie
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # ligne de hasard
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()



####################################################
print("Grid Search -  Itération des paramètres 'n_neighbors' & 'weight'")
    
# Définir le classificateur KNN
knn = neighbors.KNeighborsClassifier()

# Définir les paramètres à tester
param_grid = {
    'n_neighbors': range(5, 21),  # Tester les valeurs de k de 1 à 20
    'weights': ['uniform', 'distance'],  # Tester les poids uniformes et basés sur la distance
}

# Initialiser GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Ajuster le modèle sur l'ensemble d'entraînement
grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
print("Meilleurs paramètres : ", grid_search.best_params_)
print("Meilleure précision : ", grid_search.best_score_)
    
# Évaluer le modèle optimisé sur l'ensemble de test
best_knn = grid_search.best_estimator_
y_test_pred = best_knn.predict(X_test)

# Afficher le rapport de classification sur l'ensemble de test
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Afficher la matrice de confusion sur l'ensemble de test
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion (Ensemble de Test) - Grid Search')
plt.xlabel('Label Prédit')
plt.ylabel('Label Réel')
plt.show()

    # Écrire les résultats dans un fichier out/score.txt
if not os.path.exists('out'):  # Vérifier si le répertoire 'out' existe
    os.makedirs('out')
    
with open('out/score.txt', 'a') as f:
    f.write(f"Best Grid search parameters: {grid_search.best_params_}\n")
    f.write(f"Best Grid search accuracy: {grid_search.best_score_}\n")
    f.write(f"Classification Report for Test Set:\n{classification_report(y_test, y_test_pred)}\n")
    

print("Changement 4 : Optuna")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# Définir la fonction objective à optimiser
def objective(trial):
    try:
    # Hyperparamètres à optimiser
        k = trial.suggest_int('n_neighbors', 5, 20)  # Nombre de voisins entre 1 et 20
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Poids uniformes ou basés sur la distance
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
        leaf_size = trial.suggest_int('leaf_size', 10, 50)  # Taille des feuilles
        p = trial.suggest_int('p', 1, 2)  # Paramètre pour la distance de Minkowski (1=Manhattan, 2=Euclidean)
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithme utilisé
    
        # Création du classificateur KNN avec les paramètres suggérés
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, 
                                    leaf_size=leaf_size, p=p, algorithm=algorithm)
    
        # Évaluation du modèle avec validation croisée
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') 
        return scores.mean()  # Retourner la moyenne des scores de validation croisée
    except FloatingPointError:
        return 0 

# Définir un pruner pour interrompre les essais peu prometteurs    
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Création d'un objet d'étude
study = optuna.create_study(direction='maximize', pruner = pruner)  # Nous voulons maximiser l'accuracy
study.optimize(objective, n_trials=10, n_jobs=2)  # Effectuer 10 essais

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres : ", study.best_params)
print("Meilleure accuracy : ", study.best_value)

   # Enregistrer dans le fichier `out/score.txt`
with open('out/score.txt', 'a') as f:
    f.write(f"Best Hyperparameters Optuna1: {study.best_params}\n")
    f.write(f"Best Accuracy Optuna1: {study.best_value:.4f}\n")

# Évaluer le meilleur modèle sur l'ensemble de test
best_knn = KNeighborsClassifier(**study.best_params)
best_knn.fit(X_train, y_train)
y_test_pred = best_knn.predict(X_test)

# Afficher le rapport de classification sur l'ensemble de test
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Afficher la matrice de confusion sur l'ensemble de test
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm')
plt.title('Matrice de Confusion (Ensemble de Test) - Optuna1')
plt.xlabel('Label Prédit')
plt.ylabel('Label Réel')
plt.show()


# Visualisation de l'importance des hyperparamètres
def plot_param_importance(study):
    optuna.visualization.plot_param_importances(study).show()

# Visualisation de la distribution des hyperparamètres
def plot_param_distributions(study):
    optuna.visualization.plot_parallel_coordinate(study).show()

# Tracer la matrice de confusion pour le meilleur modèle
def plot_confusion_matrix(best_knn, X_test, y_test):
    y_pred = best_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Ensemble de Test) - Optuna 1')
    plt.xlabel('Label Prédit')
    plt.ylabel('Label Réel')
    plt.show()


fig1 = plot_param_importance(study)
plt.show(fig1)

# Plot and display parameter distributions
fig2 = plot_param_distributions(study)
plt.show(fig2)


####################################################   
print("Changement 5: Sélection de features + Optuna")


df_clean_satisfaction2=data
label_encoder = LabelEncoder()

# Encoder "Type of Travel" avec LabelEncoder
label_encoder = LabelEncoder()
df_clean_satisfaction2['Type of Travel'] = label_encoder.fit_transform(df_clean_satisfaction2['Type of Travel'])

# Encoder "Class" avec OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['Eco', 'Eco Plus','Business']])  # Spécifiez l'ordre
df_clean_satisfaction2['Class'] = ordinal_encoder.fit_transform(df_clean_satisfaction2[['Class']])

type_of_travel_classes = label_encoder.classes_

print("Mapping for 'Type of Travel':")
for index, label in enumerate(type_of_travel_classes):
    print(f"Label: '{label}' correspond à l'index: {index}")

# Accéder aux étiquettes originales et à leurs indices pour "Class"
class_categories = ordinal_encoder.categories_[0]  # Récupérer les catégories définies pour 'Class'

print("\nMapping for 'Class':")
for index, label in enumerate(class_categories):
    print(f"Label: '{label}' correspond à l'index: {index}")

X = df_clean_satisfaction2.drop(columns=['satisfaction', 'Unnamed: 0','Age','id','Departure Delay in Minutes','Arrival Delay in Minutes','Departure/Arrival time convenient'  ])
X_numeric2 = X.select_dtypes(include = ['number'])  # Caractéristiques
    
    
# Préparer les caractéristiques d'entrée (X) et la variable cible (y)
X, y = X_numeric2, y

# Normalisation des données
scaler = StandardScaler()
X_scaled2 = scaler.fit_transform(X_numeric2)

# Diviser le jeu de données une fois avant l'optimisation
X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.2, random_state=42, shuffle=True)

# Définir la fonction objective à optimiser
# Nous n'utilions cette fois-ci plus que 3 hyperparamètres (n_neighbors(k), metric et algorithm) car nous avons vu avec la précédente itération quece sont ceux qui influent le plus la performance du modèle
def objective(trial):
    try:
        # Hyperparamètres à optimiser
        k = trial.suggest_int('n_neighbors', 10, 20)  # Nombre de voisins entre 5 et 20
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithme utilisé

        # Création du classificateur KNN avec les paramètres suggérés
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric, 
                                 algorithm=algorithm)

        # Évaluation du modèle avec validation croisée
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') 
        return scores.mean()  # Retourner la moyenne des scores de validation croisée
    except FloatingPointError:
        return 0

# Définir un pruner pour interrompre les essais peu prometteurs    
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Création d'un objet d'étude
study = optuna.create_study(direction='maximize', pruner= pruner)  # Nous voulons maximiser l'accuracy
study.optimize(objective, n_trials=30 , n_jobs = 2)  # Effectuer 30 essais
   
# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres : ", study.best_params)
print("Meilleure accuracy : ", study.best_value)
        
   # Enregistrer dans le fichier `out/score.txt`
with open('out/score.txt', 'a') as f:
    f.write(f"Best Hyperparameters Optuna2: {study.best_params}\n")
    f.write(f"Best Accuracy Optuna2: {study.best_value:.4f}\n")

# Évaluer le meilleur modèle sur l'ensemble de test
best_knn = KNeighborsClassifier(**study.best_params)
best_knn.fit(X_train, y_train)
y_test_pred = best_knn.predict(X_test)

# Afficher le rapport de classification sur l'ensemble de test
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Afficher la matrice de confusion sur l'ensemble de test
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm')
plt.title('Matrice de Confusion (Ensemble de Test) - Optuna2')
plt.xlabel('Label Prédit')
plt.ylabel('Label Réel')
plt.show()
   
# Visualisation des paramètres Optuna
fig1 = optuna.visualization.plot_param_importances(study)
plt.show(fig1)
   
fig2 = optuna.visualization.plot_parallel_coordinate(study)
plt.show(fig2)

# Visualisation de l'importance des hyperparamètres
def plot_param_importance(study):
    optuna.visualization.plot_param_importances(study).show()

# Visualisation de la distribution des hyperparamètres

def plot_param_distributions(study):
    optuna.visualization.plot_parallel_coordinate(study).show()
# Tracer la matrice de confusion pour le meilleur modèle

def plot_confusion_matrix(best_knn, X_test, y_test):
    y_pred = best_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Ensemble de Test) - Optuna(features optimisés)')
    plt.xlabel('Label Prédit')
    plt.ylabel('Label Réel')
    plt.show()
 
fig1 = plot_param_importance(study)
plt.show(fig1)
# Plot and display parameter distributions
fig2 = plot_param_distributions(study)
plt.show(fig2)

######################################################################################################################
######################################################################################################################
# ---MODELE 2 : Random Forest

# Chargement de la donnée   
data = load_data()
data= data.dropna()
label_encoder = LabelEncoder()
data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
plt.title("Random forest")   

df_clean_satisfaction= data
X = df_clean_satisfaction.drop(columns=['satisfaction']) #'Unnamed: 0','id' 
X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
y = df_clean_satisfaction['satisfaction']  

df_clean_satisfaction2=data
label_encoder = LabelEncoder()

# Encoder "Type of Travel" avec LabelEncoder
label_encoder = LabelEncoder()
df_clean_satisfaction2['Type of Travel'] = label_encoder.fit_transform(df_clean_satisfaction2['Type of Travel'])

# Encoder "Class" avec OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['Eco', 'Eco Plus','Business']])  # Spécifiez l'ordre
df_clean_satisfaction2['Class'] = ordinal_encoder.fit_transform(df_clean_satisfaction2[['Class']])

# Afficher le DataFrame encodé
print(df_clean_satisfaction2.head(5))

X = df_clean_satisfaction2.drop(columns=['satisfaction', 'Unnamed: 0','Age','id','Departure Delay in Minutes','Arrival Delay in Minutes','Departure/Arrival time convenient'  ])
X_numeric2 = X.select_dtypes(include = ['number'])  # Caractéristiques
# Préparer les caractéristiques d'entrée (X) et la variable cible (y)
X, y = X_numeric2, y
# Normalisation des données
scaler = StandardScaler()
X_scaled2 = scaler.fit_transform(X_numeric2)

####################################################
print("Optuna + sélection des features - Random Forest")
# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.2, random_state=42)

# Fonction de l'objectif à minimiser (Optuna va chercher à maximiser cette fonction)
def objective(trial):
    try:# Hyperparamètres à optimiser
        n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        # Initialisation du modèle RandomForest avec les hyperparamètres choisis
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=42
        )
        
        # Utilisation de la validation croisée pour évaluer la performance du modèle
        score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()

        # Enregistrement du score dans MLflow
        #mlflow.log_metric('cross_val_accuracy', score)

        return score  # Optuna cherche à maximiser cette valeur
    except FloatingPointError:
        return 0

# Définir un pruner pour interrompre les essais peu prometteurs    
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Étude avec Optuna
study = optuna.create_study(direction='maximize', pruner = pruner)  # On cherche à maximiser l'accuracy
study.optimize(objective, n_trials=30, n_jobs=2)  # Effectuer 30 tests (trials)

# Meilleurs paramètres trouvés
print("Best parameters found: ", study.best_params)
print("Best cross-validation accuracy: ", study.best_value)
  
# Écrire l'accuracy dans un fichier
with open('out/score.txt', 'w') as f:
    f.write(f'Random Forest Accuracy on test set: {study.best_value:.4f}')
    f.write(f"Best Hyperparameters Optuna Random forest: {study.best_params}\n")

# Entraîner le modèle avec les meilleurs paramètres
best_params = study.best_params
rf_clf = RandomForestClassifier(**best_params, random_state=42)
rf_clf.fit(X_train, y_train)

# Évaluation sur l'ensemble de test
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy on test set: {accuracy:.4f}')

        # Visualisation de l'importance des hyperparamètres
def plot_param_importance(study):
    optuna.visualization.plot_param_importances(study).show()

# Visualisation de la distribution des hyperparamètres
def plot_param_distributions(study):
    optuna.visualization.plot_parallel_coordinate(study).show()

# Plot and display parameter importance
fig1 = plot_param_importance(study)
plt.show(fig1)

# Plot and display parameter distributions
fig2 = plot_param_distributions(study)
plt.show()(fig2)

######################################################################################################################
######################################################################################################################
# ---Modèle 3 : Gradient boosting
    
data = load_data()
data= data.dropna()
    
print("Gradient boosting ") 

df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
# Drop rows with missing values for 'Arrival Delay in Minutes' since it's a minor portion of the data
train_data_cleaned = df_train.dropna(subset=['Arrival Delay in Minutes'])
test_data_cleaned = df_test.dropna(subset=['Arrival Delay in Minutes'])

# Encode satisfaction (target variable) to binary
label_encoder = LabelEncoder()
train_data_cleaned['satisfaction'] = label_encoder.fit_transform(train_data_cleaned['satisfaction'])
test_data_cleaned['satisfaction'] = label_encoder.transform(test_data_cleaned['satisfaction'])

# Prendre en compte les variables categoriques:
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
train_data_cleaned = pd.get_dummies(train_data_cleaned, columns=categorical_columns)
test_data_cleaned = pd.get_dummies(test_data_cleaned, columns=categorical_columns)

# Separate features and target
X_train = train_data_cleaned.drop(columns=['satisfaction', 'Unnamed: 0', 'id'])
y_train = train_data_cleaned['satisfaction']
X_test = test_data_cleaned.drop(columns=['satisfaction', 'Unnamed: 0', 'id'])
y_test = test_data_cleaned['satisfaction']


####################################################
# Train a Gradient Boosting Classifier as a baseline model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
# Make predictions on the test set
y_pred = gbc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("accuracy",accuracy)
print("classification_rep",classification_rep)

 # Enregistrer les scores dans un fichier texte
os.makedirs('out', exist_ok=True)  # Crée le dossier 'out' s'il n'existe pas
with open('out/score.txt', 'w') as f:
    f.write(f'Accuracy on test set: {accuracy:.4f}\n')
    f.write("Classification Report:\n")
    f.write(classification_rep)


####################################################
print('XGB Booster and RandomizedSearchCV')   
# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': [3, 5, 7],
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}
 
# Initialize XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Set up RandomizedSearchCV with XGBoost
random_search_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                    n_iter=20, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

# Fit the model on the training data
random_search_xgb.fit(X_train, y_train)

# Get best parameters and the best score
best_params_xgb = random_search_xgb.best_params_
best_score_xgb = random_search_xgb.best_score_

print("best_params_xgb", best_params_xgb)
print("best_score_xgb", best_score_xgb)
    

# Enregistrer les scores dans un fichier texte
with open('out/score.txt', 'a') as file:
    file.write("Model: XGBoost Hyperparameter Tuning\n")
    file.write(f"Best Parameters: {best_params_xgb}\n")
    file.write(f"Best Cross-validation Accuracy: {best_score_xgb:.4f}\n")
    file.write("="*50 + "\n")

    
    
    
   
   
   
    
    
    
    
    
   
    
    
    
    
    
    
    
    
