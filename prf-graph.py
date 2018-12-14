import numpy as np
import os
import sys
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

def plot(plt):
  fname = sys.argv[1]
  f = open(fname,'r')
  lines = f.readlines()
  
  i = 1
  precision = []
  recall  = []
  fmeasure = []

  while i < len(lines):
      values = lines[i].split(',')
      precision.append(float(values[0]))
      recall.append(float(values[1]))
      fmeasure.append(float(values[2]))
      i += 1
  
  X = range(1,len(lines))
  
  if 'kernel' in fname:
    X = [2 ** (x-1) for x in X]
  #Output file name
  file = os.path.splitext(fname)[0] + '.png'
  

  # Plot title
  title = 'precision,recall,f1 trend on CV set'
  method = os.path.splitext(fname)[0]
  if 'kernel' in method:
    title = 'SVM with RBF kernel - ' + title
  elif 'svm' in method:
    title = 'Linear SVM - ' + title
  elif 'lr' in method:
    title = 'Logistic Regression - ' + title

  #Plot the graphs
  plt.clf()
  plt.title(title)
  plt.xlabel('Class weights for Fraud class')
  plt.plot(X, precision, linestyle = 'dashed', color = 'r', label = 'Precision')
  plt.plot(X, recall, linestyle = 'dashed', color = 'b', label = 'Recall')
  plt.plot(X, fmeasure, linestyle = 'dashed', color = 'g', label = 'F1-measure')
  plt.legend()
  plt.grid()
  plt.savefig(file)
  
plot(plt)
