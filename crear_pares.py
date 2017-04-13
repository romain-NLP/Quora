# -*-coding:Latin-1 -*
import pandas as pd
import numpy as np

import herramientas as hr

from gensim.models import word2vec

path = "/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/Datos/"
data_train = pd.read_csv(path+'train.csv',low_memory=False)#.sample(10000, random_state=23)
data_test = pd.read_csv(path+'test.csv',low_memory=False)#.sample(100000, random_state=23)

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

data_tot=pd.concat([data_train[['question1','question2','is_test']],data_test[['question1','question2','is_test']]],axis=0);


corpus = hr.build_corpus(data_tot)


#Parti pris : word2vec está entrenado sobre el conjunto limpiado
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

print "Model word2vec creado"
model.save(path+'mymodel')

#Train : Matriz que contiene los vectores promedio en cada fila
corpus_mat_train = np.zeros((2*N,100));
for i in range(2*N):
    corpus_mat_train[i,:]=hr.promedio(corpus[i],model)

#Test : Matriz que contiene los vectores promedio en cada fila
corpus_mat_test = np.zeros((2*M,100));
for i in range(2*M):
    corpus_mat_test[i,:]=hr.promedio(corpus[i+2*N],model)

del corpus, model #para aliviar la RAM

#Matriz (vec_Q1 | vec_Q2)
corpus_mat_train=np.concatenate((corpus_mat_train[:N,:],corpus_mat_train[N:,:]),axis=1)
df_vect_train=pd.DataFrame(corpus_mat_train,index=data_train.index)

#Matriz (vec_Q2 | vec_Q1)
corpus_mat_inv=np.concatenate((corpus_mat_train[:,100:],corpus_mat_train[:,:100]),axis=1)
df_vect_inv_train=pd.DataFrame(corpus_mat_inv,index=data_train.index)

#df que contienen las dos matrices anteriores
df_q_vect=pd.concat([data_train,df_vect_train],axis=1)
df_q_vect_inv=pd.concat([data_train,df_vect_inv_train],axis=1)

del data_train, corpus_mat_train, corpus_mat_inv, df_vect_train, df_vect_inv_train

df_RF_train=pd.concat([df_q_vect,df_q_vect_inv],axis=0)

del df_q_vect, df_q_vect_inv

print "Conjunto de entrenamiento creado"

#Test 
#Matriz (vec_Q1 | vec_Q2)
corpus_mat_test=np.concatenate((corpus_mat_test[:M,:],corpus_mat_test[M:,:]),axis=1)
df_vect_test=pd.DataFrame(corpus_mat_test,index=data_test.index)

df_q_vect_test=pd.concat([data_test,df_vect_test],axis=1)
print "Conjunto de test creado"
del data_test, corpus_mat_test
del df_vect_test, df_q_vect, df_q_vect_inv

df_RF_train.to_csv(path+'df_RF_train.csv',index=False)
df_q_vect_test.to_csv(path+'df_q_vect_test.csv',index=False)

