# Quora
Competition Kaggle
Pour une exploration des données, voir ici : 
https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb


3 fichiers :

crear_pares.py créé un grand data frame ou les lignes sont :
--pour l'ensemble d'entrainement (404290 paires)
<pair_id|q_id1|q_id2|q1|q2|is_duplicate|vect_q1|vect_q2> 
<pair_id|q_id1|q_id2|q1|q2|is_duplicate|vect_q2|vect_q1> pour rendre compte de la symetrie de la relation is_duplicate
--pour l'ensemble de test (2345796 paires)
<test_id|q1|q2|vect_q1|vect_q2> 
Note : le modèle word2vec a été créé en utilisant comme corpus les mots des ensembles d'entrainement et de test (légal dans la compétition).


herramientas.py stocke les méthodes utilisées dans crear_pares.py


RF_alg.py importe les dataframes créés précédemment et applique un algorithme de classification sur chaque paire
Note : Les dataframes finaux n'ont pas encore été créés faute de RAM.


A modifier : 
-créer una algorithme de classification basé sur la distance entre deux questions. Voir http://stackoverflow.com/questions/21979970/how-to-use-word2vec-to-calculate-the-similarity-distance-by-giving-2-words
-sauvegarder le word2vec.model créé (voir même lien)
-incorporer du tfidf
-matrice de confusion
