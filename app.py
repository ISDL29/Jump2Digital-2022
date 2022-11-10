# This script only generates the result.
# The full analysis can be found on Jump2Digital2022.ipynb

# IMPORTS ---------------------------------------------------------------------

import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import json

# DATA INGESTION --------------------------------------------------------------

ROOT=os.getcwd()
TRAIN_FILE=os.path.join(ROOT,'train.csv')
TEST_FILE=os.path.join(ROOT,'test.csv')
PRED_FILE=os.path.join(ROOT,'predictions.json')

train=pd.read_csv(TRAIN_FILE, sep=';')
X_train=train.drop('target',axis=1)
y_train=train['target']

X_test=pd.read_csv(TEST_FILE, sep=';')

# MODEL SETUP -----------------------------------------------------------------

# Defining the 'do nothing' transformer.
class Identity(BaseEstimator, TransformerMixin):
    '''This transformer does not make any changes to data.
    As we want to choose between dropping bad features or not in GridSearchCV, 
    we need a transformer option that does nothing to data'''
    def __init__(self):
        pass  
    def fit(self, input_array, y=None):
        return self  
    def transform(self, input_array, y=None):
        return input_array*1

# Features to drop according to the analysis on Jump2Digital2022.ipynb    
dropfeats=['feature7','feature8']

# Do nothing transformer
identity=Identity()

# Drop bad features transformer
drop=ColumnTransformer(transformers=[('dropfts','drop',dropfeats)],
                       remainder='passthrough')
rf=RandomForestClassifier(random_state=2022)

# Pipeline setup.
# 1. Preprocessing: Drop features or not. Default option is doing nothing.
# 2. Random forest classifier.
pipeline=Pipeline([('pre',identity),('rf',rf)])

# Model grid
# 1. Whether drop bad features or not.
# 2. Random forest hyperparameters options.
pipe_grid={'pre':[identity,drop],
           'rf__n_estimators':[100,150,200],
           'rf__min_samples_split':[2,3,4,5],
           'rf__bootstrap':[True,False],
           'rf__class_weight':['balanced',None],
           }

# Model initialization.
clf=GridSearchCV(pipeline,param_grid=pipe_grid,verbose=0,cv=2)

# PREDICTION ------------------------------------------------------------------

# Model training and prediction
clf.fit(X_train,y_train)
y_test=pd.DataFrame(clf.predict(X_test),columns=['target'])

# Saving predictions in a json file.
y_test.to_json(PRED_FILE,indent=2)