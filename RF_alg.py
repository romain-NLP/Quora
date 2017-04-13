# -*-coding:Latin-1 -*

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

path = "/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/Datos/"
df_RF_train = pd.read_csv(path+'df_RF_train.csv',low_memory=False)
df_q_vect_test = pd.read_csv(path+'df_q_vect_test.csv',low_memory=False)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100,n_jobs=-1,verbose=1) 
print "RF inicializado"

#Cross-validation
kf = cross_validation.KFold(len(df_RF_train), n_folds=10)
cm=np.zeros((2,2));
for train_index, test_index in kf:

   X_train, X_test = df_RF_train.iloc[train_index,7:], df_RF_train.iloc[test_index,7:]
   y_train, y_test = df_RF_train.loc[train_index,'is_duplicate'], df_RF_train.loc[test_index,'is_duplicate']

   forest.fit(X_train, y_train)
   cm+=confusion_matrix(y_test, forest.predict(X_test))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('title'+'.png')


class_names=['not_dup','dup']
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


#SUBMISSION / ENTREGA

#  = forest.fit(df_RF_train.ix[:,7:], df_RF_train["is_duplicate"])
# scores = cross_validation.cross_val_score(forest, df_RF_train.ix[:,7:], df_RF_train["is_duplicate"], cv=10,n_jobs=-1)
# Take the mean of the scores (because we have one for each fold)
# print(scores.mean())

# predictions = forest.predict(df_q_vect_test.ix[:,4:])
# submission = pandas.DataFrame({
#        "is_duplicate": predictions,
#        "test_id": df_q_vect_test["test_id"]
#    })

# submission.to_csv(path+'sub_RF_1.csv',index=False)