# CS-229-project

This page documents steps on how to run python scripts related to my CS 229 project. The steps are many - and some scripts make take hours to run on original data. The dataset is more than 400 MB - and github does not allow me to check in those files to this repo.

1. Create a conda environment using project.yml
2. Download 'PaySim dataset' - https://www.kaggle.com/ntnu-testimon/paysim1 and save the csv as 'all.csv' file.
3. Run `python3 data-extraction.py all.csv [output_dir]` - This script takes in the entire PaySim dataset and creates different csv files for each transaction type: `transfer.csv`, `cash_out.csv`, `cash_in.csv`, `debit.csv`, `payment.csv`. If `output_dir` is provided and doesn't exist, it will be created.
4. Run `python3 pca.py <path/to>/transfer.csv` - This script takes `transfer.csv`, performs PCA in 2D and saves the plot to `./pca_results/transfer.png` (it creates `pca_results/` if needed). If the CSV path includes directories, only the basename is used for the plot title and png name.
5. Run `python3 logreg.py <path/to>/transfer.csv` - This file takes in transfer dataset and runs logistic regression with a fast class-weight sweep (10 iterations, `cw[1]=2**it`). It writes precision/recall/f1 per iteration to `./prf/lr_transfer_prf.txt` and saves models to `./models/lr_transfer_<classweight>.sav`. For the deep sweep, run `python3 logreg.py <path/to>/transfer.csv --sweep linear --max-iters 513`.
6. Run `python3 svm.py <path/to>/transfer.csv` - This file runs a linear SVM (`LinearSVC`) with a fast class-weight sweep (10 iterations, `cw[1]=2**it`). It writes precision/recall/f1 per iteration to `./prf/svm_transfer_prf.txt` and saves models to `./models/svm_transfer_<classweight>.sav`. For the deep sweep, run `python3 svm.py <path/to>/transfer.csv --sweep linear --max-iters 513`. Same experiment can be repeated with `cash_out.csv` as well.
7. Run 'python kernel_svm.py transfer.csv'. Same experiment can be repeated with 'cash_out.csv' as well.

Experiments from steps 5,6,7 take a very long time to run - anywhere between 30 mins to 28 hours !!

1. Run 'python prf-graph.py <input_file>' - <input_file> can be any file created in prf/ directory. This script plots figures similar to those in fig.2 and fig. 3 - to show precision, recall and f1-measure trends for changing class weights of fraud samples
2. Run 'python test.py <filename> <mode>' - This file runs all three algorithms - LR,SVM and SVM with RBF kernel for selected class weights. <filename> can be 'transfer.csv' or 'cash_out.csv'. 'mode' can be 'train' , 'validation' or 'test'. The script produces precision-recall curves, precision,recall and AUPRC values are mentioned in 'Results' section of the paper.

