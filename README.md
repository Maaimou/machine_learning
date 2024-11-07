### Prédiction de la Satisfaction des Passagers d'Avion
Contexte et Objectif
Objectif : Prédire la satisfaction des passagers aériens.

### Enjeux
1. Identifier les facteurs de satisfaction
But : Identifier les variables qui influencent la satisfaction (ponctualité, confort, service, etc.).
Impact : Prioriser les actions pour améliorer l'expérience client.

2. Prédire la mécontentement
But : Identifier les passagers susceptibles d’être insatisfaits.
Impact : Permettre des actions correctives pour réduire les avis négatifs et fidéliser les clients.

3. Segmenter les passagers selon leur satisfaction
But : Regrouper les passagers selon leurs attentes.
Impact : Offrir des services personnalisés et optimiser l'expérience par segment.

### Description du Projet
Nous développons un modèle de machine learning pour classifier les passagers comme satisfaits ou insatisfaits en fonction de leurs caractéristiques démographiques et de leur évaluation des services de vol. Le jeu de données comprend 129 880 lignes et 25 colonnes, incluant :

**Caractéristiques démographiques** : 'Sexe', 'type de client', 'âge', 'type de voyage', 'classe'.

**Évaluations des services** : 'Wi-Fi en vol', 'confort du siège', 'divertissement',..., notés de 1 à 5.

**Retards de vol** : 'Retards au départ et à l’arrivée'.

**Cible** : 'Satisfaction' (satisfait ou non).

### Approche
#### 1. Installation et Importation des Bibliothèques
**Librairies utilisées** : pandas, numpy, seaborn, matplotlib pour la manipulation de données et visualisation.
**Machine Learning**: scikit-learn.
**Optimisation** : optuna pour le réglage des hyperparamètres.
**Analyse exploratoire** : sweetviz pour un rapport général des données.

#### 2. Chargement et Exploration des Données
Identification des valeurs manquantes (393 lignes supprimées).
Analyse des distributions et corrélations avec la satisfaction.
Facteurs clés : Embarquement en ligne, confort du siège, divertissement en vol, type de voyage, classe.
Un rapport (rapport_EDA.html) donne un aperçu global du jeu de données.

#### 3. Modèle 1 : KNN Classifier 
	
**Pourquoi ce modèle ?** 
Ce modèle est un algorithme de classification facile d'interprétation qui classe les données en fonction des k voisins les plus proches dans l'espace des caractéristiques. Dans notre  cas de la satisfaction des passagers, cela signifie que le modèle classe chaque passager en fonction des passagers qui lui ressemblent le plus, en se basant sur les évaluations et caractéristiques démographiques.
		
De plus, ce modèle est capable de prendre en compte correctement plusieurs dimensions de donnnées, cela nous permettra de capturer les relations entre différentes caractéristiques.
	
a. *Baseline** 
	
Pour notre Baseline, nous utilisons : 
	- X composé que des caractéristiques numériques de notre dataset
	- Y notre variable cible
  - KNeighborsClassifier() : C'est un modèle de classification qui se base sur l'algorithme des k-plus proches voisins (KNN). Le principe de KNN est de prédire la classe d'un échantillon en fonction des classes des "k" voisins les plus proches. Ce modèle prend les caractéristiques (données) d'entraînement X_train et les étiquettes associées y_train pour apprendre à prédire les classes.
 - knn.fit(X_train, y_train) : Cette méthode entraîne le modèle en ajustant ses paramètres (par exemple, les distances entre les points et les voisins) en utilisant les données d'entraînement X_train et les labels y_train. Le modèle apprend à identifier la relation entre les caractéristiques (données) et les classes cibles pour pouvoir prédire les classes de nouveaux échantillons.
	- Test_train_split pour séparer nos données de test et d'entrainement
 
b. *Normalisation de la donnée** 

**Pourquoi ?** 
Le scaling des données est crucial avec l'utilisation de KNN, car il harmonise les échelles des caractéristiques. Sans scaling, les caractéristiques avec des valeurs plus grandes influencent davantage les calculs de distance, pouvant biaiser le modèle. Le scaling permet de donner une importance égale à chaque caractéristique, améliorant la performance et la fiabilité des modèles de machine learning qui s’appuient sur les distances pour prendre leurs décisions

StandardScaler() : soustrait la moyenne de chaque caractéristique (colonne) et divise par son écart-type. Cela donne des données avec une moyenne de 0 et une variance de 1. 

fit_transform(X_numeric) : Cette méthode calcule la moyenne et l'écart-type des données numériques présentes dans X_numeric (les caractéristiques du jeu de données), puis applique la transformation pour obtenir X_scaled, où chaque caractéristique est standardisée.

train_test_split(X_scaled, y, test_size=0.2, random_state=42) : Cette fonction découpe le jeu de données en un ensemble d'entraînement (80%) et un ensemble de test (20%).
X_scaled : Ce sont les caractéristiques des données normalisées.

test_size=0.2 : Cela signifie que 20% des données seront utilisées pour tester le modèle, et les 80% restants seront utilisés pour l'entraînement.
random_state=42 : Ce paramètre garantit la reproductibilité des résultats. Si vous utilisez la même valeur pour random_state, vous obtiendrez toujours le même découpage du jeu de données.






