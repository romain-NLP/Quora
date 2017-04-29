# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import functions_preproc as fn_pp
from scipy import spatial
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import log_loss as loss
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score as CV
from sklearn.feature_extraction.text import TfidfVectorizer

import logging as log
log.basicConfig(level=log.DEBUG)
#import gensim
#import nltk
#import ...

#path_model = '/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/word2vec_models/'
#model = gensim.models.Word2Vec.load(path_model+'model_text8')


class Model:
    def __init__(self, trainDataset, N_train, testDataset=None, model = LogReg):
        self.OurModel = model() #RF, SVM, Reglog
        self.FeatureFunctions = [self.feat1,self.feat2,self.feat3]
        self.FeatureName = ["dist", "dist_pond", "name3"]
        self.trainData = pd.read_csv(trainDataset, low_memory=False).sample(N_train, random_state=23)
        self.testData = testDataset if testDataset is not None else pd.DataFrame()
        self.loadFeatures()


    def saveFeatures(self, name='features.csv', test=False):
        self.Features.to_csv(name)
        if test:
            self.testFeatures.to_csv('test'+name)


    def loadFeatures(self, name='features.csv',test=True):
        try:
            self.Features = pd.read_csv(name, index_col=0)
        except:
            self.Features = pd.DataFrame(self.trainData.ix[:,'is_duplicate'])
        self.testFeatures = pd.DataFrame()
        if test:
            try:
                self.testFeatures = pd.read_csv('test'+name, index_col=0)
            except:
                pass




    def ComputeFeatures(self, test = False, features= [0,1,2]):
        '''
        Compute selected features, replace in existing (?) self.Features
        :param features: 
        :return: 
        '''
        if not test:
            for f in features:
                log.info('Computing features %s ...'%f)
                #j'ai changé trainData[['question1','question2']] pour trainData
                applyTemp = self.trainData.apply(self.FeatureFunctions[f], axis=1)
                log.info('Done !')
                if isinstance(applyTemp, pd.Series):
                    self.Features[self.FeatureName[f]] = applyTemp
                else:
                    for col in applyTemp.columns:
                        self.Features[self.FeatureName[f] + col] = applyTemp[col]
        else:
            for f in features:
                log.info('Computing features %s ...' % f)
                applyTemp = self.trainData.apply(self.FeatureFunctions[f], axis=1)
                log.info('Done !')
                if isinstance(applyTemp, pd.Series):
                    self.testFeatures[self.FeatureName[f]] = applyTemp
                else:
                    for col in applyTemp.columns:
                        self.testFeatures[self.FeatureName[f] + col] = applyTemp[col]


    def Train(self,trainDataX,trainDataY):
        '''
        :param trainDataX: part of self.features on which we want to train the data
        :return: 
        '''
        self.OurModel = self.OurModel.fit(trainDataX,trainDataY)


    def Test(self, testDataX):
        #Test model. Return probability. If true is given , return metrics (kaggle metrics, F1-score, tableau )
        res = self.OurModel.predict_proba(testDataX)
        return res


    def CrossValidate(self, cv=10):
        Metrics = {}
        Metrics['accuracy'] = make_scorer(accuracy_score)
        Metrics['F1'] = make_scorer(f1_score)
        Metrics['Precision'] = make_scorer(precision_score)
        Metrics['Recall']= make_scorer(recall_score)
        Metrics['Loss']= make_scorer(loss, greater_is_better=False, needs_proba=True)
        for metric in Metrics:
            scores = CV(self.OurModel, self.Features.ix[:, self.Features.columns != 'is_duplicate'],
                          self.Features.is_duplicate, cv=cv,scoring=Metrics[metric])
            log.info(metric + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    

    def ChangeModel(self, model=LogReg):
        self.OurModel = model() #Changer RegLog en random forest/SVM

    def Preprocessing(self,N_train):
        import functions_preproc as fn_pp
        #Preprocess string function
        self.trainData = self.trainData.dropna(how="any") 
        self.trainData['question1_mod'] = pd.Series(dtype=object)
        self.trainData['question1_mod'] = pd.Series(dtype=object)
        for col,col_mod in zip(['question1', 'question2'],['question1_mod', 'question2_mod']):
            self.trainData[col_mod] = self.trainData[col].apply(fn_pp.clean_sentence)
        corpus = [];
        tfidf_list_q1 = [None] * N_train;
        tfidf_list_q2 = [None] * N_train;

        for col in ['question1', 'question2']:
            for sentence in self.trainData[col].iteritems():
                corpus.append(sentence[1])
        tf = TfidfVectorizer(input='content', analyzer='word', ngram_range=(0,1),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)
        tfidf_matrix = tf.fit_transform(corpus)
        feature_names = tf.get_feature_names()
        del corpus 
        print('corpus borrado')
        for i in range(N_train):
            tfidf_list_q1[i] = fn_pp.tfidf_weights(i,tfidf_matrix,feature_names)
            tfidf_list_q2[i] = fn_pp.tfidf_weights(N_train + i,tfidf_matrix,feature_names)
    
        print('listas tfidf creadas')
        mat_id = self.trainData['id'].values
    
        self.trainData['dict_tfidf_q1'] = pd.Series(np.array(tfidf_list_q1), index = mat_id)
        self.trainData['dict_tfidf_q2'] = pd.Series(np.array(tfidf_list_q2), index = mat_id)
        print('Series tfidf creados')
        del tfidf_matrix, feature_names

    #Nos features
    #@staticmethod
    def feat1(self,strs):
    	import functions_preproc as fn_pp
        """Fonction de distance w2v entre deux vecteurs moyens non pondérés"""
        q1 = fn_pp.prom_preg(self.trainData['question1_mod'], self.trainData['dict_tfidf_q1'], pond = False);
        q2 = fn_pp.prom_preg(self.trainData['question2_mod'], self.trainData['dict_tfidf_q2'], pond = False);    
        return fn_pp.distancia(q1,q2)

    #@staticmethod
    def feat2(self):
    	import functions_preproc as fn_pp
        """Fonction de distance w2v entre deux vecteurs moyens oui pondérés"""
        q1 = fn_pp.prom_preg(self.trainData['question1_mod'], self.trainData['dict_tfidf_q1'], pond = True);
        q2 = fn_pp.prom_preg(self.trainData['question2_mod'], self.trainData['dict_tfidf_q2'], pond = True);    
        return fn_pp.distancia(q1,q2)

        

    @staticmethod
    def feat3(strs):
        return 0




if __name__ == "__main__":
    TrainSet = pd.read_csv('train.csv',index_col = 0)
    TestSet = pd.read_csv('test.csv',index_col = 0)
    MonModele = Model(TrainSet,N_train,TestSet)
