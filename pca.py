import numpy as np
import pandas as pd
import csv
import sys
import os

# Run:
#   python3 pca.py path/to/transactions.csv
# Output:
#   Saves PCA scatter plot to ./pca_results/<csv_basename>.png

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA

def compute_pca(plt):
   filename = sys.argv[1]
   base = os.path.splitext(os.path.basename(filename))[0]
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

   print("Number of transactions " , len(y))
   
   scaler = preprocessing.StandardScaler()
   x = scaler.fit_transform(x)

   pca = PCA(n_components=2)
   pcs = pca.fit_transform(x)
   pdf = pd.DataFrame(data = pcs,columns = ['PC 1', 'PC 2'])

   x = pdf.values.tolist()
   x = [tuple(l) for l in x]
   y = y.tolist()
   
   #print('List of labels')
   #print(y)

   be = []
   fr = []
   i = 0
   while i < len(y):
        if y[i][0] == 0:
       	    be.append(x[i])
        else:
       	    fr.append(x[i])
        i += 1
   
   plt.clf()
   plt.title('PCA for ' + base + ' transactions')
   plt.xlabel('PC 1')
   plt.ylabel('PC 2 ')
   plt.grid(True)


   x1,x2 = zip(*be)
   plt.scatter(x1, x2,color = 'b' , marker='+' , label = 'Non-fraud txn')
   
   if len(fr) > 0:
   	   x1,x2 = zip(*fr)
   	   plt.scatter(x1, x2,color = 'r' , marker = 'x' , label = 'Fraud txn')

   fig = base
   plt.legend()
   if not os.path.exists('pca_results'):
       os.makedirs('pca_results')
   plt.savefig('./pca_results/' + fig +'.png')



compute_pca(plt)
