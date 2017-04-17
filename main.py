import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import log_loss as loss
#import ...


class Model:
    def __init__(self, trainDataset, testDataset=None, model = LogReg):
        self.OurModel= model() #RF, SVM, Reglog
        self.FeatureFunctions = [self.feat1,self.feat2,self.feat3]
        self.FeatureName = ["name1", "name2", "name3"]
        self.trainData = trainDataset
        self.testData=testDataset if testDataset else pd.DataFrame()
        self.loadFeatures()





    def saveFeatures(self, name='features.csv'):
        self.Features.to_csv(name)

    def loadFeatures(self, name='features.csv'):
        try:
            self.Features = pd.read_csv(name, index_col=0)
        except:
            self.Features = pd.DataFrame()


    def ComputeFeatures(self, train = True, features= [1,2,3], save=False):
        '''
        Compute selected features, replace in existing (?) self.Features
        :param features: 
        :return: 
        '''
        dataset = self.trainData if train else self.testData


        for f in features:
            self.Features[self.FeatureName[f]] = dataset.apply(self.FeatureFunctions[f], axis=1)

        if save:
            self.saveFeatures()


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
        return 0

    def feat2(self):
        return 0

    def feat3(self):
        return 0



if __name__ == "__main__":
    TrainSet = pd.read_csv('train.csv')
    TestSet = pd.read_csv('test.csv')
    MonModele = Model(TrainSet,TestSet)

