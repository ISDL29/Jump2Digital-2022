import os
import pandas as pd

ROOT=os.getcwd()
TRAIN_FILE=os.path.join(ROOT,'train.csv')
TEST_FILE=os.path.join(ROOT,'test.csv')

train=pd.read_csv(TRAIN_FILE, sep=';')
test=pd.read_csv(TEST_FILE, sep=';')

print(train)