#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# MINIBATCH_saving.py                                                                           #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/07/2020                                                                     #
# ============================================================================================= #
"""
Creates the DECAGON Minibatch object which contains the node-node interactions categorized into 
their respective edge type. It also separates the whole dataset into training, validation and 
test datasets, generating at the same time datasets of negative interations. The object is saved
into a pickle python2 readable format.

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
batch_size : int
    Number of training examples in each batch. Defaults to 512.
val_test_size : float
    Fraction of the dataset that will form the validation and test dataset. It has to be lower 
    than 0.5. Defaults to 0.15.
"""
# ============================================================================================= #
from __future__ import print_function
import argparse
import os
import warnings
import time
import datetime
import numpy as np
import pickle
import psutil
from decagon.deep.minibatch import EdgeMinibatchIterator

# FILE PATHS
parser = argparse.ArgumentParser(description='MINIBATCH building')
parser.add_argument('in_file', type=str, help="Input DECAGON file")
parser.add_argument('batch_size', type=int, default=512, nargs='?',\
                    help = 'Number of training instances in a train batch.')
parser.add_argument('val_test_size', type=float, default=0.15, nargs='?',\
                    help='Fraction of test and validation.')
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
# BEGIN
start = time.time()
pid = os.getpid()
ps= psutil.Process(pid)
warnings.filterwarnings("ignore")
# ============================================================================================= #
# LOAD DATA STRUCTURE
print('\n==== IMPORTED VARIABLES ====')
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
print('\n')
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
# ============================================================================================= #
# CREATE MINIBATCH
print("Create minibatch iterator\n")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=args.batch_size,
    val_test_size=args.val_test_size
)
# ============================================================================================= #
# EXPORT DATA
out_file = 'data/data_structures/MINIBATCH/MINIBATCH_' + words[2]+\
            '_genes_' + str(n_genes) + '_drugs_'+ str(n_drugs) + '_se_' + str(n_se_combo)+\
            '_batchsize_'+str(args.batch_size)+'_valsize_'+str(args.val_test_size)
print('Output file: ',out_file,'\n')
memUse = ps.memory_info()
data = {}
data['minibatch'] = minibatch
data['mb_vms'] = memUse.vms
data['mb_rss'] = memUse.rss
data['mb_time'] = time.time()-start
with open(out_file,'wb') as f:
    pickle.dump(data, f, protocol=2)
memUse = ps.memory_info()
print('Minibatch created!')
print('Virtual memory: ', memUse.vms*1e-09,'Gb')
print('RSS Memory: ', memUse.rss*1e-09,'Gb')
print('Total time: ', datetime.timedelta(seconds=time.time()-start))
