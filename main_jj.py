# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import log_loss as loss
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score as CV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV as GS_CV
from sklearn.feature_extraction.text import TfidfVectorizer
import functions_preproc as fn_pp
from functools import reduce
from nltk.corpus import stopwords
import logging as log
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(message)s')
import re
#import ...

#path_model = '/Users/Romain/Documents/UChile/Matematicas/Semestre_4/Memoria/Problemas/Quora/word2vec_models/'
#model = gensim.models.Word2Vec.load(path_model+'model_text8')


class Model:
    def __init__(self, trainDataset,  testDataset=None, N_train=False,model = LogReg):
        self.OurModel = model() #RF, SVM, Reglog
        self.FeatureFunctions = [self.feat1,self.feat2,self.feat3,self.feat4,self.feat5,self.feat6,self.feat7]
        self.FeatureName = ["sim", "sim_pond", "common_words","common_words_ratio","sentence_length_difference","normed_diff_pond","normed_diff"]
        if N_train:
            self.trainData = trainDataset.sample(N_train, random_state=23)
        else:
            self.trainData = trainDataset
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




    def ComputeFeatures(self, test = False, features= None):
        '''
        Compute selected features, replace in existing (?) self.Features
        :param features: 
        :return: 
        '''
        if features is None:
            features = range(len(self.FeatureFunctions))
        if not test:
            for f in features:
                log.debug('Computing features %s ...'%f)
                applyTemp = self.trainData.apply(self.FeatureFunctions[f], axis=1)
                log.debug('Done !')
                if isinstance(applyTemp, pd.Series):
                    log.debug(self.FeatureName[f])
                    self.Features[self.FeatureName[f]] = applyTemp
                else:
                    for col in applyTemp.columns:
                        self.Features[self.FeatureName[f] + col] = applyTemp[col]
                        log.debug(self.FeatureName[f]+col)
        else:
            for f in features:
                log.debug('Computing features %s ...' % f)
                applyTemp = self.trainData.apply(self.FeatureFunctions[f], axis=1)
                log.debug('Done !')
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

#NOUVEAU YOUHOUUU
    def CV_predict(self, trainDataX = None,trainDataY = None):
    	#Attention le test est le train ici !
    	trainDataX = self.Features.ix[:, self.Features.columns != 'is_duplicate'] if trainDataX is None else trainDataX
        trainDataY = self.Features.is_duplicate if trainDataY is None else trainDataY
        res = cross_val_predict(self.OurModel, trainDataX, trainDataY, cv=10, n_jobs=-1, fit_params=None, method='predict')
        return res

#NOUVEAU YOUHOUUU
    def gridsearch(self,param_grid,trainDataX=None, trainDataY=None):
    	trainDataX = self.Features.ix[:, self.Features.columns != 'is_duplicate'] if trainDataX is None else trainDataX
        trainDataY = self.Features.is_duplicate if trainDataY is None else trainDataY
        
    	Metrics = {}
    	Metrics['Loss'] = make_scorer(loss, greater_is_better=False, needs_proba=True)
    	clf = GS_CV(self.OurModel, param_grid, scoring=Metrics['Loss'], fit_params=None, n_jobs=-1, refit=True, cv=10)
    	clf.fit(trainDataX,trainDataY)
    	print("Returns three elements : results, best_param and the scorer used")
    	return pd.DataFrame(clf.cv_results_), clf.best_estimator_, clf.scorer_


    def CrossValidate(self,trainDataX=None, trainDataY=None,  cv=10):
        trainDataX = self.Features.ix[:, self.Features.columns != 'is_duplicate'] if trainDataX is None else trainDataX
        trainDataY = self.Features.is_duplicate if trainDataY is None else trainDataY
        Metrics = {}
        Metrics['accuracy'] = make_scorer(accuracy_score)
        Metrics['F1'] = make_scorer(f1_score)
        Metrics['Precision'] = make_scorer(precision_score)
        Metrics['Recall'] = make_scorer(recall_score)
        Metrics['Loss'] = make_scorer(loss, greater_is_better=False, needs_proba=True)
        for metric in Metrics:
            scores = CV(self.OurModel, trainDataX, trainDataY, cv=cv, scoring=Metrics[metric])
            log.info(metric + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    

    def ChangeModel(self, model=LogReg):
        self.OurModel = model() #Changer RegLog en random forest/SVM
        

    def Preprocessing(self):
        #Preprocess string function
        self.trainData = self.trainData.dropna(how="any")
        log.debug("CLEANING SENTENCES...")
        STOP_WORDS = set(stopwords.words())
        regex = re.compile('([^\s\w]|_)+')
        for col,col_mod in zip(['question1', 'question2'],['question1_mod', 'question2_mod']):
            self.trainData[col_mod] = self.trainData[col].apply(fn_pp.clean_sentence,args=[STOP_WORDS,regex])
        log.debug("DONE !")

        log.debug("FILLING CORPUS...")
        corpus = self.trainData['question1'].tolist() + self.trainData['question2'].tolist()
        log.debug("DONE ! Corpus : %s"%corpus[:10])

        log.debug("TRAINING TFIDF ...")
        tfidf_list_q1 = {}
        tfidf_list_q2 = {}
        tf = TfidfVectorizer(input='content', analyzer='word', ngram_range=(0,1),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)
        tfidf_matrix = tf.fit_transform(corpus)
        feature_names = tf.get_feature_names()
        for i,index in enumerate(self.trainData.index):
            tfidf_list_q1[index] = fn_pp.tfidf_weights(i, tfidf_matrix, feature_names)
            tfidf_list_q2[index] = fn_pp.tfidf_weights(i+len(self.trainData), tfidf_matrix, feature_names)
        log.debug("DONE ! Liste : %s"%tfidf_list_q1.items()[:10])
        self.trainData['dict_tfidf_q1'] = pd.Series(tfidf_list_q1)
        self.trainData['dict_tfidf_q2'] = pd.Series(tfidf_list_q2)

    #Nos features
    #@staticmethod
    def feat1(self,row):
        """Fonction de distance w2v entre deux vecteurs moyens non pondérés"""
        q1 = fn_pp.prom_preg(row['question1_mod'], row['dict_tfidf_q1'], pond = False);
        q2 = fn_pp.prom_preg(row['question2_mod'], row['dict_tfidf_q2'], pond = False);
        return fn_pp.distancia(q1,q2)

    def feat2(self,row):
        """Fonction de distance w2v entre deux vecteurs moyens oui pondérés"""
        q1 = fn_pp.prom_preg(row['question1_mod'], row['dict_tfidf_q1'], pond = True);
        q2 = fn_pp.prom_preg(row['question2_mod'], row['dict_tfidf_q2'], pond = True);
        return fn_pp.distancia(q1, q2)


    @staticmethod
    def feat3(row):
        '''
        :return: Number of Common Words
        '''
        word_list1 = set(row.question1_mod)
        word_list2 = set(row.question2_mod)
        feat = len(word_list1.intersection(word_list2))
        if feat !=feat:
            print row
        return feat

    def feat4(self,row):
        '''
        :param q1: 
        :param q2: 
        :return: Ratio of common words
        '''
        word_list1 = set(row.question1_mod)
        word_list2 = set(row.question2_mod)
        feat = float(len(word_list1.intersection(word_list2))) / len(word_list1.union(word_list2)) if len(word_list1.union(word_list2)) != 0 else 0
        if feat !=feat:
            print row
        return feat#**0.21


    def feat5(self,row):
        '''
        :param q1: 
        :param q2: 
        :return: words length difference
        '''
        feat = np.abs(len(row.question2_mod) - len(row.question1_mod))
        if feat !=feat:
            print row
        return feat

#NOUVEAU YOUHOUU
    def feat6(self,row):
        """Fonction de distance w2v entre deux vecteurs moyens oui pondérés"""
        q1 = fn_pp.prom_preg(row['question1_mod'], row['dict_tfidf_q1'], pond = True);
        q1 = q1/np.linalg.norm(q1) if np.linalg.norm(q1) != 0 else q1
        q2 = fn_pp.prom_preg(row['question2_mod'], row['dict_tfidf_q2'], pond = True);
        q2 = q2/np.linalg.norm(q2) if np.linalg.norm(q2) != 0 else q2
        return np.linalg.norm(q1 - q2)

#NOUVEAU YOUHOUU
    def feat7(self,row):
        """Fonction de distance w2v entre deux vecteurs moyens oui pondérés"""
        q1 = fn_pp.prom_preg(row['question1_mod'], row['dict_tfidf_q1'], pond = False);
        q1 = q1/np.linalg.norm(q1) if np.linalg.norm(q1) != 0 else q1
        q2 = fn_pp.prom_preg(row['question2_mod'], row['dict_tfidf_q2'], pond = False);
        q2 = q2/np.linalg.norm(q2) if np.linalg.norm(q2) != 0 else q2
        return np.linalg.norm(q1 - q2)

if __name__ == "__main__":
    TrainSet = pd.read_csv('train.csv',index_col = 0)
    TestSet = pd.read_csv('test.csv',index_col = 0)
    MonModele = Model(TrainSet, TestSet, 100000)
    MonModele.Preprocessing()
    MonModele.ComputeFeatures()
