# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import spatial
import numpy as np
import re
import nltk
import gensim
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_sentence(val,STOP_WORDS = set(nltk.corpus.stopwords.words()),regex=re.compile('([^\s\w]|_)+')):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    new_sentence = filter(lambda w: w not in STOP_WORDS, sentence)
    return new_sentence

def tfidf_weights(doc,tfidf_matrix,feature_names):
    dict_w={};
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        dict_w[w]=s
    return dict_w

path_model = 'w2v'
model = gensim.models.Word2Vec.load(join(path_model, 'model_text8'))

def prom_preg(data,dict_w,pond):
    prom_q = np.zeros(100)
    denum = 0
  
    if pond == True: 
        for word in data:
            if word in model.wv and word in dict_w:
                prom_q += dict_w[word]*model.wv[word]
                denum += dict_w[word]

    if pond == False: 
        for word in data:
            if word in model.wv:
                prom_q += model.wv[word]
                denum += 1
                
    if denum != 0:
        prom_q = prom_q/denum
    return prom_q

def distancia(val1,val2):
    result = 1 - spatial.distance.cosine(val1,val2)

    return result if result==result else -1


