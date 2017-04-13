# -*-coding:Latin-1 -*

import pandas as pd
import numpy as np

print "Training the random forest..."
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

path = "/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/Datos/"
df_RF_train = pd.read_csv(path+'df_RF_train.csv',low_memory=False)
df_q_vect_test = pd.read_csv(path+'df_q_vect_test.csv',low_memory=False)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

print "RF inicializado"

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(df_RF_train.ix[:,7:], df_RF_train["is_duplicate"])
scores = cross_validation.cross_val_score(forest, df_RF_train.ix[:,7:], df_RF_train["is_duplicate"], cv=10)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

#predictions = forest.predict(df_q_vect_test.ix[:,4:])
#submission = pandas.DataFrame({
#        "is_duplicate": predictions,
 #       "test_id": df_q_vect_test["test_id"]
  #  })

#submission.to_csv(path+'sub_RF_1.csv',index=False)