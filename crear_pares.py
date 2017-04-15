# -*-coding:Latin-1 -*
import pandas as pd
import numpy as np
import herramientas as hr

from gensim.models import word2vec


path = "/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/Datos/"
data_train = pd.read_csv(path+'train.csv',low_memory=False).sample(1000, random_state=23)
data_test = pd.read_csv(path+'test.csv',low_memory=False).sample(1000, random_state=23)
print "Datos importados"


data_train = hr.clean_dataframe(data_train)
data_test = hr.clean_dataframe(data_test)
print "Datos limpiados"


N=len(data_train);
M=len(data_test);
is_test_train=np.zeros(N,dtype='int');
is_test_test=np.ones(M,dtype='int');
data_train['is_test']=pd.Series(is_test_train,index=data_train.index);
data_test['is_test']=pd.Series(is_test_test,index=data_test.index);
data_train_test=pd.concat([data_train[['question1','question2','is_test']],data_test[['question1','question2','is_test']]],axis=0);
corpus = hr.build_corpus(data_train_test)
print "Corpus creado"


#Parti pris : word2vec está entrenado sobre el conjunto limpiado
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model.save(path+'mymodel')
print "Model word2vec creado y guardado"



del corpus #hint para aliviar la RAM

df_RF_train = hr.clean_dataframe_after_building_model(data_train)
df_RF_test = hr.clean_dataframe_after_building_model(data_test)

del model 

df_RF_train.to_csv(path+'df_RF_train.csv',index=False)
df_RF_test.to_csv(path+'df_RF_test.csv',index=False)

