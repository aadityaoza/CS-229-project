# CS-229-project: PaySim fraud detection

## Project description
This repo contains scripts for a CS229 project on **fraud detection** using the **PaySim** transactions dataset. It trains and evaluates models under severe class imbalance by sweeping the fraud class weight and reporting precision/recall/F1 and PR curves.

Models included:
- Logistic Regression (`logreg.py`)
- Linear SVM (`svm.py`, `LinearSVC`)
- RBF / kernel SVM (`kernel_svm.py`, `SVC`)

## Setup
Create the conda environment:

```bash
conda env create -f project.yml
conda activate project
```

## Data
Download the PaySim dataset from Kaggle and save it anywhere (filename can be anything, e.g. `syn.csv` or `all.csv`):
`https://www.kaggle.com/ntnu-testimon/paysim1`

These scripts expect the standard PaySim header:
`step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`

## Quickstart (recommended pipeline)

### 1) Split the full dataset by transaction type
This creates `transfer.csv`, `cash_out.csv`, `cash_in.csv`, `debit.csv`, `payment.csv`.

```bash
python3 data-extraction.py syn.csv data/
```

### 2) PCA visualization (2D)
Saves to `./pca_results/<csv_basename>.png`.

```bash
python3 pca.py data/transfer.csv
```

### 3) Train sweeps (fast vs deep)
All scripts accept paths like `data/transfer.csv` and use only the basename (`transfer`) for naming outputs.

#### Logistic Regression

```bash
python3 logreg.py data/transfer.csv
python3 logreg.py data/transfer.csv --sweep linear --max-iters 513
```

#### Linear SVM (`LinearSVC`)

```bash
python3 svm.py data/transfer.csv
python3 svm.py data/transfer.csv --sweep linear --max-iters 513
```

#### RBF / kernel SVM (`SVC`)

```bash
python3 kernel_svm.py data/transfer.csv
python3 kernel_svm.py data/transfer.csv --sweep linear --max-iters 513
python3 kernel_svm.py data/transfer.csv --weight 32
```

## Outputs
- `models/`: saved trained models (`*.sav`), e.g. `lr_transfer_32.sav`, `svm_transfer_32.sav`, `kernel_svm_transfer_32.sav`
- `prf/`: precision/recall/F1 logs per class weight and related plots
- `pca_results/`: PCA scatter plots
- `test-results/`, `validation/`: outputs produced by `test.py` and other evaluation scripts

## Plot PRF trends
Plot precision/recall/F1 trends vs class weight:

```bash
python3 prf-graph.py prf/lr_transfer_prf.txt
```

## Final evaluation / PR curves
Runs LR + Linear SVM + RBF SVM at selected class weights.

```bash
python3 test.py transfer.csv validation
```

## Runtime note
Deep sweeps (e.g. 513 weights) can take a long time (tens of minutes to many hours depending on dataset size and machine).

