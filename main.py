# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import log_loss as loss
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score as CV
import logging as log
log.basicConfig(level=log.DEBUG)
#import ...


class Model:
    def __init__(self, trainDataset, testDataset=None, model = LogReg):
        self.OurModel= model() #RF, SVM, Reglog
        self.FeatureFunctions = [self.feat1,self.feat2,self.feat3]
        self.FeatureName = ["name1", "name2", "name3"]
        self.trainData = trainDataset
        self.testData=testDataset if testDataset is not None else pd.DataFrame()
        self.loadFeatures()


    def saveFeatures(self, name='features.csv', test=False):
        self.Features.to_csv(name)
        if test:
            self.testFeatures.to_csv('test'+name)


    def loadFeatures(self, name='features.csv',test=True):
        try:
            self.Features = pd.read_csv(name, index_col=0)
        except:
            self.Features = pd.DataFrame(self.trainData.is_duplicate)
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
                applyTemp = self.trainData[['question1','question2']].apply(self.FeatureFunctions[f], axis=1)
                log.info('Done !')
                if isinstance(applyTemp, pd.Series):
                    self.Features[self.FeatureName[f]] = applyTemp
                else:
                    for col in applyTemp.columns:
                        self.Features[self.FeatureName[f] + col] = applyTemp[col]
        else:
            for f in features:
                log.info('Computing features %s ...' % f)
                applyTemp = self.trainData[['question1','question2']].apply(self.FeatureFunctions[f], axis=1)
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

    def Preprocessing(self,string):
        #Preprocess string function
        pass

    #Nos features
    @staticmethod
    def feat1(strs):
        return pd.Series([0,0])

    @staticmethod
    def feat2(strs):
        return 0

    @staticmethod
    def feat3(strs):
        return 0



if __name__ == "__main__":
    TrainSet = pd.read_csv('train.csv',index_col = 0)
    TestSet = pd.read_csv('test.csv',index_col = 0)
    MonModele = Model(TrainSet,TestSet)

