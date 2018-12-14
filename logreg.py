import numpy as np
import pandas as pd
import csv
import sys
import os

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import warnings
import pickle

max_iters = 10
n_estimators = 10

def logreg(x,y,filename):

   # Model output file name
   file = (os.path.splitext(filename))[0]
   fname = './models/lr_' + file +'/'

   # File for writing precision,recall, f-measure scores for fraud transactions
   f = open('./prf/lr_'+ file + '_prf' +'.txt' ,'w')
   f.write('precision,recall,f-score \n')

   # Stratified sampling based on Y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Create 15% validation set and 15% test set split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)
   
   #Iterations
   it = 0
   
   # Run training algorithm for multiple class weights
   while it < max_iters:
       cw = {}
       cw[0] = 1
       cw[1] = 2 ** it
       # Train
       print('**************************************')
       print("Iteration number  " , it)
       lr = LogisticRegression(class_weight = cw)
       print('Class weights ', cw)
       lr.fit(X_train,y_train)

       # Save trained model to disk
       name = fname + str(cw[1]) + '.sav'
       pickle.dump(lr, open(name, 'wb'))

       # Predict on validation data
       y_val_pred = lr.predict(X_val)
       print('Performance on validation data - Confusion matrix')
       print(confusion_matrix(y_val,y_val_pred))
   
       precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
       print('Precision, Recall, F-score, Support  on validation data' )
       print("Precision" , precision)
       print("Recall" , recall)
       print("F-score" , fscore)
       print("Support" , support)

       p1 = precision[1]
       r1 = recall[1]
       f1 = fscore[1]

       f.write(str(p1) +','+ str(r1) + ',' + str(f1) + '\n') 
       it += 1

   f.close()

def run():
   filename = sys.argv[1]
   df = pd.read_csv(filename, usecols = [2,4,5,7,8,9] , header = 0,
   	names = ['Amount','Source-OB','Source-NB','Dest-OB','Dest-NB','target'])
   
   results = list(map(int, df['target'])) 
   print('Number of fraudulent transactions ' , sum(results))

   features = ['Amount', 'Source-OB', 'Source-NB', 'Dest-OB' , 'Dest-NB']
   targets = ['target']

   # Separating out the features and target variables
   x = df.loc[:, features].values
   y = df.loc[:, targets].values

   y  = [i for j in y for i in j]
   
   #Ignore warnings
   warnings.filterwarnings("ignore", category=FutureWarning)

   print("***********Logistic Regression**********")
   logreg(x,y,filename)
  
run()
