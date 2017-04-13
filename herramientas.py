# -*-coding:Latin-1 -*

import pandas as pd
import numpy as np
import re
import nltk

from gensim.models import word2vec


STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

#Función transformando las preguntas en una lista de listas de palabras
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

#Fonción computando el vector promedio de una lista de palabra
def promedio(list_corp,model):
    prom = np.zeros(100)
    i=0;
    for word in list_corp:
        if word in model.wv:
            i+=1;
            prom += model.wv[word]
    if i!= 0:
        prom=prom/i;
    return prom
            