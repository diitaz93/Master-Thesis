#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# store_minibatch.py                                                                            #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/07/2020                                                                     #
# ============================================================================================= #
"""
Creates the minibatch containing the edges of the dataset and saves it in a pickle readable file.

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
"""
# ============================================================================================= #
from __future__ import print_function
import argparse
import os
import time
import datetime
import tensorflow as tf
import numpy as np
import pickle
import psutil
from decagon.deep.minibatch import EdgeMinibatchIterator

# FILE PATHS
parser = argparse.ArgumentParser(description='Train DEGAGON')
parser.add_argument('in_file',type=str, help="Input file with data structures")
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
add_str = ''
# BEGIN
start = time.time()
pid = os.getpid()
ps= psutil.Process(pid)
# ============================================================================================= #
# LOAD DATA STRUCTURE
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
print(n_genes,n_drugs,n_se_combo,n_se_mono,DSE)
val_test_size = 0.15
batch_size = 512
# ============================================================================================= #
# CREATE MINIBATCH
print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=batch_size,
    val_test_size=val_test_size
)
# ============================================================================================= #
# EXPORT DATA
out_file = 'data/data_structures/MINIBATCH/MINIBATCH_'+ add_str + words[2]+\
            '_genes_' + str(n_genes) + '_drugs_'+ str(n_drugs) + '_se_' + str(n_se_combo)+\
            '_batchsize_'+str(batch_size)+'_valsize_'+str(val_test_size)
print(out_file)
memUse = ps.memory_info()
data = {}
data['minibatch'] = minibatch
data['mb_vms'] = memUse.vms
data['mb_rss'] = memUse.rss
data['mb_time'] = time.time()-start
with open(out_file,'wb') as f:
    pickle.dump(data, f, protocol=2)
memUse = ps.memory_info()
print('Virtual memory:', memUse.vms*1e-09,'Gb')
print('RSS Memory:', memUse.rss*1e-09,'Gb')
print('Total time:', datetime.timedelta(seconds=time.time()-start))
