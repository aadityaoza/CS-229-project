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
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import warnings
import pickle

transfer_weights = [70,39,16]
cash_out_weights = [145,132,128]

lr_prc = [0]*2
svm_prc = [0]*2
kernel_svm_prc = [0]*2

def plot_pr_curve(filename,type = 'validation'):
   from sklearn.metrics import f1_score
   from sklearn.metrics import auc
   from sklearn.metrics import average_precision_score
   
   print('************* Precision Recall metrics on ' + type + ' data *************')
   plt.clf()

   # Logistic Regression
   pre_lr , rec_lr, thresh_lr = precision_recall_curve(lr_prc[0], lr_prc[1],pos_label = 1)
   area = auc(rec_lr, pre_lr)
   area = round(area,4)
   print('Logistic Regression - Area under PRC' , area)
   plt.plot(rec_lr, pre_lr, linestyle='--' , color = 'r',label = 'Logistic Regression - AUPRC - '  + str(area))


   # Linear SVM
   pre_svm , rec_svm, thresh_svm = precision_recall_curve(svm_prc[0], svm_prc[1],pos_label = 1)
   area = auc(rec_svm, pre_svm)
   area = round(area,4)
   print('Linear SVM - Area under PRC' , area)
   plt.plot(rec_svm, pre_svm, linestyle='--' , color = 'b',label = 'Linear SVM - AUPRC - '  + str(area))


   # RBF SVM
   pre_kernel_svm , rec_kernel_svm, thresh_kernel_svm = precision_recall_curve(kernel_svm_prc[0], kernel_svm_prc[1],pos_label = 1 )
   area = auc(rec_kernel_svm, pre_kernel_svm)
   area = round(area,4)

   print('RBF Precision ' , pre_kernel_svm)
   print('RBF Recall ' , rec_kernel_svm)
   
   print('SVM with RBF kernel - Area under PRC' , area)
   plt.plot(rec_kernel_svm, pre_kernel_svm, linestyle='--' , color = 'g',label = 'SVM RBF kernel - AUPRC - '  + str(area))
   
   
   if "transfer" in filename:
       plt.title('TRANSFER  - Precision-Recall curve')
   elif "cash" in filename:
       plt.title('CASH OUT  - Precision-Recall curve')

   # plot the precision recall curve for the model
   plt.xlabel('Recall')
   plt.ylabel('Precision')
   plt.legend()
   # show the plot
   plt.savefig('./test-results/'+filename)

   print('*********************************************************************')

def print_confusion_matrix(cm,mode = 'validation' , dataset = 'Transfer', algo = 'lr'):
    return

def lr_test(x,y,filename,mode = 'validation'):
   # Stratified sampling based on Y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Create 15% validation set and 15% test set split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)

   # Load trained model from saved models
   dataset = ''
   model_file = ''
   if "transfer" in filename:
      model_file = './models/lr_transfer/' + str(transfer_weights[0]) +'.sav'
      dataset = 'Transfer'
   if "cash_out" in filename:
      model_file = './models/lr_cash_out/' + str(cash_out_weights[0]) +'.sav'
      dataset = 'Cash Out'
      
   lr = pickle.load(open(model_file, 'rb'))
   print('Class weights ', lr.class_weight)
   print('Feature importance ', lr.coef_)


   # Predict on train data
   if mode == 'train':
     y_train_pred_prob = lr.decision_function(X_train)
     lr_prc[0] = y_train
     lr_prc[1] = y_train_pred_prob
   
     y_train_pred = lr.predict(X_train)
     print('Performance on train data - Confusion matrix')
     cm = confusion_matrix(y_train,y_train_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'lr')

     precision,recall,fscore,support=score(y_train,y_train_pred,average=None)
     print('Precision, Recall, F-score, Support on Train data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)

   # Predict on validation data
   if mode == 'validation':
     y_val_pred_prob = lr.decision_function(X_val)
     lr_prc[0] = y_val
     lr_prc[1] = y_val_pred_prob
   
     y_val_pred = lr.predict(X_val)
     print('Performance on validation data - Confusion matrix')
     cm = confusion_matrix(y_val,y_val_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'lr')

     precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
     print('Precision, Recall, F-score, Support on validation data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)

   # Predict on test data
   if mode == 'test':
      y_test_pred_prob = lr.decision_function(X_test)
      lr_prc[0] = y_test
      lr_prc[1] = y_test_pred_prob

      y_test_pred = lr.predict(X_test)
      print('Performance on test data - Confusion matrix')
      cm = confusion_matrix(y_test,y_test_pred)
      print(cm)
      print_confusion_matrix(cm,mode,dataset)


      precision,recall,fscore,support=score(y_test,y_test_pred,average=None)
      print('Precision, Recall, F-score, Support on test data' )
      print("Precision" , precision)
      print("Recall" , recall)
      print("F-score" , fscore)
      print("Support" , support)
   
def svm_test(x,y,filename,mode = 'validation'):
   # Stratified sampling based on Y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Create 15% validation set and 15% test set split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)

   # Load trained model from saved models
   dataset = ''
   model_file = ''
   if "transfer" in filename:
      model_file = './models/svm_transfer/' + str(transfer_weights[1]) +'.sav'
      dataset = 'Transfer'
   if "cash_out" in filename:
      model_file = './models/svm_cash_out/' + str(cash_out_weights[1]) +'.sav'
      dataset = 'Cash Out'
      
   svm = pickle.load(open(model_file, 'rb'))
   print('Class weights ', svm.class_weight)
   print('Feature importance ', svm.coef_)

   # Predict on train data
   if mode == 'train':
     y_train_pred_prob = svm.decision_function(X_train)
     svm_prc[0] = y_train
     svm_prc[1] = y_train_pred_prob
   
     y_train_pred = svm.predict(X_train)
     print('Performance on train data - Confusion matrix')
     cm = confusion_matrix(y_train,y_train_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'svm')

     precision,recall,fscore,support=score(y_train,y_train_pred,average=None)
     print('Precision, Recall, F-score, Support on Train data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)
   
   # Predict on validation data
   if mode == 'validation':
     y_val_pred_prob = svm.decision_function(X_val)
     svm_prc[0] = y_val
     svm_prc[1] = y_val_pred_prob
   
     y_val_pred = svm.predict(X_val)
     print('Performance on validation data - Confusion matrix')
     cm = confusion_matrix(y_val,y_val_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'svm')


     precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
     print('Precision, Recall, F-score, Support on validation data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)

   # Predict on test data
   if mode == 'test':
      y_test_pred_prob = svm.decision_function(X_test)
      svm_prc[0] = y_test
      svm_prc[1] = y_test_pred_prob

      y_test_pred = svm.predict(X_test)
      print('Performance on test data - Confusion matrix')
      cm = confusion_matrix(y_test,y_test_pred)
      print(cm)
      print_confusion_matrix(cm,mode,dataset)

      precision,recall,fscore,support=score(y_test,y_test_pred,average=None)
      print('Precision, Recall, F-score, Support on test data' )
      print("Precision" , precision)
      print("Recall" , recall)
      print("F-score" , fscore)
      print("Support" , support)

def kernel_svm_test(x,y,filename,mode = 'validation'):
   # Stratified sampling based on Y
   X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y , test_size=0.30, random_state=42)

   # Create 15% validation set and 15% test set split
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,stratify=y_test , test_size=0.50, random_state=42)

   # Load trained model from saved models
   dataset = ''
   model_file = ''
   if "transfer" in filename:
      model_file = './models/kernel_svm_transfer/' + str(transfer_weights[2]) +'.sav'
      dataset = 'Transfer'
   if "cash_out" in filename:
      model_file = './models/kernel_svm_cash_out/' + str(cash_out_weights[2]) +'.sav'
      dataset = 'Cash Out'
      
   svm = pickle.load(open(model_file, 'rb'))
   print('Class weights ', svm.class_weight)
   
   # Predict on train data

   if mode == 'train':
     y_train_pred_prob = svm.decision_function(X_train)
     kernel_svm_prc[0] = y_train
     kernel_svm_prc[1] = y_train_pred_prob
   
     y_train_pred = svm.predict(X_train)
     print('Performance on train data - Confusion matrix')
     cm = confusion_matrix(y_train,y_train_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'kernel_svm')

     precision,recall,fscore,support=score(y_train,y_train_pred,average=None)
     print('Precision, Recall, F-score, Support on Train data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)
   
   # Predict on validation data
   if mode == 'validation':
     y_val_pred_prob = svm.decision_function(X_val)
     kernel_svm_prc[0] = y_val
     kernel_svm_prc[1] = y_val_pred_prob
   
     y_val_pred = svm.predict(X_val)
     print('Performance on validation data - Confusion matrix')
     cm = confusion_matrix(y_val,y_val_pred)
     print(cm)
     print_confusion_matrix(cm,mode,dataset,'kernel_svm')

     precision,recall,fscore,support=score(y_val,y_val_pred,average=None)
     print('Precision, Recall, F-score, Support on validation data' )
     print("Precision" , precision)
     print("Recall" , recall)
     print("F-score" , fscore)
     print("Support" , support)

   # Predict on test data
   if mode == 'test':
      y_test_pred_prob = svm.decision_function(X_test)
      kernel_svm_prc[0] = y_test
      kernel_svm_prc[1] = y_test_pred_prob

      y_test_pred = svm.predict(X_test)
      print('Performance on test data - Confusion matrix')
      cm = confusion_matrix(y_test,y_test_pred)
      print(cm)
      print_confusion_matrix(cm,mode,dataset)

      precision,recall,fscore,support=score(y_test,y_test_pred,average=None)
      print('Precision, Recall, F-score, Support on test data' )
      print("Precision" , precision)
      print("Recall" , recall)
      print("F-score" , fscore)
      print("Support" , support)
      
      
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

   mode = sys.argv[2]

   print("**************** Logistic Regression Test *******************")
   lr_test(x,y,filename,mode)

   print("**************** SVM Test *******************")
   svm_test(x,y,filename,mode)

   print("**************** Kernel SVM Test *******************")
   kernel_svm_test(x,y,filename,mode)

   if "transfer" in filename:
       plot_pr_curve('transfer'+'_'+mode+'.png' , mode)
   elif "cash" in filename:
       plot_pr_curve('cash_out'+'_'+mode+'.png' , mode)

   
  
run()
