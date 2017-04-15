# -*-coding:Latin-1 -*

import pandas as pd
import numpy as np
import re
import nltk
import gensim

from gensim.models import word2vec


STOP_WORDS = nltk.corpus.stopwords.words()

# Transforma una frase en una lista de palabras de importancia
# (palabra de importancia = palabra != STOPWORDS)
def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    sentence = re.sub("[^\w]", " ",  sentence).split()
    return sentence


# Aplica el metodo clean_sentence a cada pregunta en las columnas
# 'question1' y 'question2'
def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['question1', 'question2']:
        data[col] = data[col].astype(object)
        data[col] = data[col].apply(clean_sentence)
    
    return data


# Función transformando las preguntas en una lista de listas de palabras
# que pueda servir de entreda en word2vec
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1]#.split(" ")
            corpus.append(word_list)
            
    return corpus


#Fonción computando el vector promedio de una lista de palabra
def promedio(list_corp):
    prom = np.zeros(100)
    i=0;
    for word in list_corp:
        if word in model.wv:
            i+=1;
            prom += model.wv[word]
    if i!= 0:
        prom=prom/i;
    return prom

path = "/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/Datos/"
model = gensim.models.Word2Vec.load(path+'mymodel')

# incluye en el df las coordenadas del vector promedio de cada pregunta
# y duplica el dataframe (inicialmente <vect_q1|vect_q2|is_dup> )
# con el orden :  <vect_q2|vect_q1|is_dup>
# y concatena el df duplicado bajo el original
def clean_dataframe_after_building_model(data):

    
    data['q1_vect']=pd.Series(dtype=object)
    data['q2_vect']=pd.Series(dtype=object)
    
    for col,col_vect in zip(['question1', 'question2'],['q1_vect','q2_vect']):
        data[col_vect] = data[col].apply(promedio) #doute : argument 'model' de promedio ?
    
    index=data.index
    columns = range(200)
    data_bis=pd.DataFrame(index=index,columns=columns)
    
    for i in data.index:
        for j in range(100):
            data_bis.loc[i,j]=data.loc[i,'q1_vect'][j]
            data_bis.loc[i,j+100]=data.loc[i,'q2_vect'][j]
    
    data = pd.concat([data,data_bis],axis=1)
        
    data = data.drop('q1_vect', 1)
    data = data.drop('q2_vect', 1)
    
    data.rename(columns = lambda x: str(x), inplace=True)
    #condición verificada solo para el cjto de entrenamiento
    if 'is_duplicate' in data:
        data_inv=pd.DataFrame(columns=data.columns)#,index=data.index)
        #data_inv['id']=data['id']
    
        data_inv['is_duplicate']=data['is_duplicate'].values
        data_inv.loc[:,'0':'99'] = data.loc[:,'100':'199'].values
        data_inv.loc[:,'100':'199'] = data.loc[:,'0':'99'].values

        data = pd.concat([data,data_inv],axis=0)
    
    return data
            