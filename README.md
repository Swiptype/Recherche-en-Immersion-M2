Différentes versions en python pour implémenter un système complet de détection d'anomalies sur du texte (Text Anomaly Detection) en utilisant une approche d'apprentissage profond non supervisé (ou semi-supervisé). 

Le but est de créer un modèle capable d'apprendre à reconnaître une catégorie de texte "normale" (par exemple, des articles sur le sport) et de rejeter tout autre type de texte (politique, religion, informatique, etc.)
comme étant une "anomalie", sans jamais avoir vu ces anomalies durant l'entraînement.

Plusieurs ajouts on été faits au fur et à mesure des avancées du projet :

Il inclut notamment également un module d'explicabilité pour visualiser pourquoi un texte est considéré comme une anomalie.

=================================================================================================================================================================================================================================

ACTUEL : recherche_v5.py

Architecture du Modèle : Deep SVDD

L'Encodeur (DistilBERT) : Le script utilise DistilBertModel, une version allégée de BERT, pour transformer les phrases en vecteurs numériques (embeddings).

La Projection : Les vecteurs sortant de BERT passent par un petit réseau de neurones (self.projection) qui réduit la dimension et nettoie l'information.

Le Concept du Centre c : L'idée du SVDD est de trouver une sphère compacte qui englobe toutes les données "normales".

  Le modèle apprend à projeter les textes normaux le plus près possible d'un centre c prédéfini.

  Score d'anomalie : C'est la distance euclidienne au carré entre la représentation du texte et le centre c.
  
  Plus le score est élevé, plus le texte est considéré comme une anomalie.


=======


A. Préparation des Données (TextDataset)

Le script utilise le jeu de données classique 20 Newsgroups.

Nettoyage Spécifique : On reitre les "stop words" (mots courants comme "the", "and") avant de donner le texte au modèle pour éviter que ces mots vides ne soient considérés comme la cause de l'anomalie ("rationales").


B. Entraînement (train_and_eval)

  Initialisation du Centre : Avant l'entraînement, le script passe quelques données dans le modèle non entraîné et calcule la moyenne des sorties pour fixer le centre c.

  Apprentissage : Il entraîne le modèle à minimiser la distance entre les textes de la classe normale et ce centre c.

    Note technique : Il gèle les poids de DistilBERT au début, puis les débloque (UNFREEZE_AFTER_EPOCH) pour un ajustement fin (fine-tuning).

  Évaluation : Il teste le modèle sur un mélange de toutes les catégories. Le but est de voir s'il attribue des scores bas à la catégorie normale et hauts aux autres (calcul de l'AUC ROC).


C. Explicabilité (IGExplainer)

Une fois qu'une anomalie est détectée, le script utilise Integrated Gradients (librairie captum) pour expliquer pourquoi.

  Le script calcule l'importance de chaque mot dans la décision du modèle.

  Visualisation Console : Il utilise colorama pour afficher le texte dans le terminal.

    Rouge vif : Mots qui contribuent fortement à l'anomalie.

    Gris : Mots ignorés ou peu importants.


D. Exécution (main)

Le script lance un benchmark complet :

  Il boucle sur chacune des 20 catégories du dataset.

  Pour chaque catégorie, il considère qu'elle est la "normale" et les 19 autres sont des anomalies.

  Il répète l'expérience 5 fois pour avoir des statistiques fiables (moyenne et écart-type de l'AUC).

  Il sauvegarde les résultats dans benchmark_results.csv.


=======

EXPLICATION DES RÉSULTATS DE "benchmark_results_v5.csv"


1. Category (Le sujet normal)

C'est la catégorie que le modèle a vue pendant l'entraînement. Il a appris à quoi ressemble un texte de ce sujet spécifique (ex: comp.graphics).


2. AUC_mean (La performance moyenne)

Elle représente l'Aire Sous la Courbe ROC (AUC), moyennée sur les 5 essais.

  Échelle : De 0.5 (Hasard total) à 1.0 (Perfection).

  Interprétation :

    Proche de 1.0 (Excellent) : Le sujet est très distinctif. Le modèle le sépare très facilement du reste.

    Exemple : comp.sys.ibm.pc.hardware est à 0.799. Bon score, le vocabulaire technique aide probablement à l'isoler.

    Proche de 0.5 - 0.6 (Moyen/Difficile) : Le sujet est vague ou ressemble trop aux autres catégories.

    Exemple : alt.atheism est à 0.649. C'est plus faible. Les discussions sur l'athéisme partagent beaucoup de vocabulaire avec les catégories religieuses (soc.religion.christian, etc.), 
              ce qui crée de la confusion pour le modèle qui considère la religion comme une anomalie.


3. AUC_std (La stabilité)

C'est l'écart-type sur les 5 essais. Cela mesure la fiabilité de l'entraînement.

    Valeur faible (ex: 0.006) : Le modèle est stable. En relançant l'entraînement, le résultat sera le même. comp.os.ms-windows.misc (0.0068) est très stable.

    Valeur élevée (ex: > 0.02) : Le résultat varie beaucoup d'un entraînement à l'autre. Cela peut indiquer que le modèle a du mal à converger ou qu'il est sensible à l'initialisation aléatoire.




















  
