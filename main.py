import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import log_loss as loss
#import ...


class Model:
    def __init__(self, trainDataset, testDataset=pd.DataFrame(), model = LogReg):
        self.OurModel= model() #RF, SVM, Reglog
        pass

    def ComputeFeatures(self, features= [1,2,3]):
        '''
        Compute selected features, replace in existing (?) self.Features
        :param features: 
        :return: 
        '''
        featurefunction = [self.feat1,self.feat2,self.feat3]
        self.Features = pd.DataFrame()
        pass

    def saveFeatures(self, name='features.csv'):
        self.Features.to_csv(name)

    def loadFeatures(self, name='features.csv'):
        pass

    def ApplyPreprocessing(self):
        #Preprocess string function
        pass

    def Train(self,trainData):
        # Train model with selected features
        pass

    def Test(self, test, truth=None):
        #Test model. Return probability. If true is given , return metrics (kaggle metrics, F1-score, tableau )
        pass

    def CrossValidate(self):
        # Met ta library ici
        pass


    def ChangeModel(self, model=LogReg):
        self.OurModel = model() #Changer RegLog en random forest/SVM

    #Nos features
    def feat1(self):
        pass

    def feat2(self):
        pass

    def feat3(self):
        pass
