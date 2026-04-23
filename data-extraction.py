"""
Splits an input CSV into separate CSV files by transaction type.
Run: python3 data-extraction.py syn.csv output_dir
"""

import numpy as np
import math
import pandas as pd
import csv
import os
import sys

def syn_cc(filename  = 'syn.csv', out_dir = ''):
    f = open(filename)
    # read first line
    header = f.readline()
    types = set(['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT'])

    path = out_dir
    if out_dir != '' and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if path != '' and not path.endswith('/'):
        path = path + '/'
    cash_out_data = path + 'cash_out.csv'
    transfer_data = path + 'transfer.csv'
    payment_data = path + 'payment.csv'
    cash_in_data = path + 'cash_in.csv'
    debit_data = path + 'debit.csv'
    
    co = open(cash_out_data,'w') 
    tr = open(transfer_data,'w')
    py = open(payment_data,'w')
    ci = open(cash_in_data,'w')
    db = open(debit_data,'w')
    
    co.write(header)
    tr.write(header)
    py.write(header)
    ci.write(header)
    db.write(header)

    # Read all lines in syn.csv
    # Create new csv files for each transaction type
    lines = f.readlines()[1:]

    for line in lines:
        entries  = line.split(',')
        type = entries[1]
        if type  == 'CASH_OUT' :
            co.write(line)
        elif type == 'TRANSFER':
            tr.write(line)
        elif type == 'PAYMENT':
            py.write(line)
        elif type == 'CASH_IN':
            ci.write(line)
        elif type == 'DEBIT':
            db.write(line)

    co.close()
    tr.close()
    py.close()
    ci.close()
    db.close()
    f.close()

    co = open(cash_out_data,'r') 
    tr = open(transfer_data,'r')
    py = open(payment_data,'r')
    ci = open(cash_in_data,'r')
    db = open(debit_data,'r')

syn_cc(sys.argv[1] if len(sys.argv) > 1 else 'syn.csv', sys.argv[2] if len(sys.argv) > 2 else '')

