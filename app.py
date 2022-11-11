# Este código solo genera la solución.
# El análisis completo se encuentra en Jump2Digital2022.ipynb

# IMPORTACIONES ---------------------------------------------------------------

import os
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# INGESTA DE DATOS ------------------------------------------------------------

ROOT=os.getcwd()
TRAIN_FILE=os.path.join(ROOT,'train.csv')
TEST_FILE=os.path.join(ROOT,'test.csv')
PRED_JSON_FILE=os.path.join(ROOT,'predictions.json')
PRED_CSV_FILE=os.path.join(ROOT,'predictions.csv')
CV_RES_FILE=os.path.join(ROOT,'cv_results.csv')

train=pd.read_csv(TRAIN_FILE, sep=';')
X_train=train.drop('target',axis=1)
y_train=train['target']

X_test=pd.read_csv(TEST_FILE, sep=';')

# CONFIGURACIÓN DEL MODELO ----------------------------------------------------

# Definiendo el transformador neutro.
class Identity(BaseEstimator, TransformerMixin):
    '''Este transformador deja intacto los datos. Su utilidad es proporcionar
    una opción en la que no se eliminan las características 7 y 8 en 
    GridSearchCV.'''
    def __init__(self):
        pass  
    def fit(self, input_array, y=None):
        return self  
    def transform(self, input_array, y=None):
        return input_array*1

# Característica a eliminar según el análisis en Jump2Digital2022.ipynb    
dropfeats=['feature7','feature8']

# Transformador neutro
identity=Identity()

# Transformador que elimina las características 7 y 8
drop=ColumnTransformer(transformers=[('dropfts','drop',dropfeats)],
                       remainder='passthrough')
rf=RandomForestClassifier(random_state=2022)

# Configuración del pipeline
# 1. Preprocesamiento: No tocar los datos
# 2. Clasificador random forest
pipeline=Pipeline([('pre',identity),('rf',rf)])

# Opciones de configuración del modelo
# 1. Eliminación de características 7 y 8 o no hacer nada
# 2. Hiperparámetros del random forests a explorar
pipe_grid={'pre':[identity,drop],
           'rf__n_estimators':[100,150,200],
           'rf__min_samples_split':[2,3,4,5],
           'rf__bootstrap':[True,False],
           'rf__class_weight':['balanced',None],
           }

# Inicialización del modelo
clf=GridSearchCV(pipeline,
                 param_grid=pipe_grid,
                 return_train_score=True,
                 verbose=0,
                 cv=7
                 )

# PREDICCIÓN ------------------------------------------------------------------

# Entrenamiento del modelo y predicción
clf.fit(X_train,y_train)
y_test=pd.DataFrame(clf.predict(X_test),columns=['target'])

# Guardado de los resultados de la exploración de hiperparámetros
clf.cv_results_.to_csv(CV_RES_FILE)

# Guardado de las predicciones en formato json y csv
y_test.to_json(PRED_JSON_FILE,indent=2)
y_test.to_csv(PRED_CSV_FILE,index=False)

# =============================================================================