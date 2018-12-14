# CS-229-project
This page documents steps on how to run python scripts related to my CS 229 project. The steps are many - and some scripts make take hours to run on original data. The dataset is more than 400 MB - and github does not allow me to check in those files to this repo.

1) Create a conda environment using project.yml
2) Download 'PaySim dataset' - https://www.kaggle.com/ntnu-testimon/paysim1 and save the csv as 'all.csv' file.
3) Run 'python syn_cc.py all.csv' - This script takes in entire PaySim dataset and creates different csv files for each transaction type - transfer.csv, cash_out.csv, cash_in.csv, debit.csv,payment.csv
4) Run 'python pca.py transfer.csv' - This script takes transfer.csv, performs PCA on it in 2d and plots the result in transfer.png file. These results are present in pca-results. It can also be run with 
5) Run 'python logreg-deep.py transfer.csv' - This file takes in transfer dataset, and runs logistic regression for 512 different class weight combinations. At each iteration, it computes precision,recall and f1-measure and writes it to a file in prf/ folder. Same experiment can be repeated with 'cash_out.csv' as well.
6) Run 'python svm-deep.py transfer.csv' . Same experiment can be repeated with 'cash_out.csv' as well.
7) Run 'python kernel_svm.py transfer.csv'. Same experiment can be repeated with 'cash_out.csv' as well.

Experiments from steps 5,6,7 take a very long time to run - anywhere between 30 mins to 28 hours !!

8) Run 'python prf-graph.py <input_file>' - <input_file> can be any file created in prf/ directory. This script plots figures similar to those in fig.2 and fig. 3 - to show precision, recall and f1-measure trends for changing class weights of fraud samples

9) Run 'python test.py <filename> <mode>' - This file runs all three algorithms - LR,SVM and SVM with RBF kernel for selected class weights. <filename> can be 'transfer.csv' or 'cash_out.csv'. 'mode' can be 'train' , 'validation' or 'test'. The script produces precision-recall curves, precision,recall and AUPRC values are mentioned in 'Results' section of the paper.
  

