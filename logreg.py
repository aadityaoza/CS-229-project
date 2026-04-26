import numpy as np
import pandas as pd
import csv
import sys
import os
import argparse

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

n_estimators = 10

def iter_class_weights(sweep, start_it, max_iters):
   it = start_it
   while it < max_iters:
       cw = {}
       cw[0] = 1
       if sweep == 'linear':
           cw[1] = it
       else:
           cw[1] = 2 ** it
       yield it, cw
       it += 1

def logreg(x,y,filename,sweep='exp',start_it=0,max_iters=10):

   # Model output file name
   file = os.path.splitext(os.path.basename(filename))[0]
   
   if not os.path.exists('./models'):
       os.makedirs('./models')
   if not os.path.exists('./prf'):
       os.makedirs('./prf')

   fname = './models/lr_' + file +'_'

   # File for writing precision,recall, f-measure scores for fraud transactions
   f = open('./prf/lr_'+ file + '_prf' +'.txt' ,'w')
   f.write('precision,recall,f-score \n')

   # Stratified sampling based on Y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Create 15% validation set and 15% test set split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)
   
   # Run training algorithm for multiple class weights
   for it, cw in iter_class_weights(sweep, start_it, max_iters):
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
   parser = argparse.ArgumentParser()
   parser.add_argument('csv_file')
   parser.add_argument('--sweep', choices=['exp','linear'], default='exp')
   parser.add_argument('--max-iters', type=int, default=None)
   parser.add_argument('--start-it', type=int, default=None)
   args = parser.parse_args()

   filename = args.csv_file
   # PaySim CSV header:
   # usecols indices [2,4,5,7,8,9] correspond to:
   # amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
   df = pd.read_csv(filename, usecols = [2,4,5,7,8,9] , header = 0,
   	names = ['Amount','Source-OB','Source-NB','Dest-OB','Dest-NB','target'])
   
   results = list(map(int, df['target'])) 
   print('Number of fraudulent transactions ' , sum(results))

   features = ['Amount', 'Source-OB', 'Source-NB', 'Dest-OB' , 'Dest-NB']
   targets = ['target']

   # Separating out the features and target variables
   x = df.loc[:, features].values
   y = df.loc[:, targets].values

   # Convert 2D array to 1D array
   y  = [i for j in y for i in j]
   
   #Ignore warnings
   warnings.filterwarnings("ignore", category=FutureWarning)

   print("***********Logistic Regression**********")
   max_iters = args.max_iters
   start_it = args.start_it
   if max_iters is None:
       max_iters = 513 if args.sweep == 'linear' else 10
   if start_it is None:
       start_it = 1 if args.sweep == 'linear' else 0
   logreg(x,y,filename,args.sweep,start_it,max_iters)
  
run()
