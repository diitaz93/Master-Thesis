#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import time
import datetime
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pickle
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model_red import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.deep.layers import dropout_sparse
from decagon.utility import rank_metrics, preprocessing

# FILE PATHS
parser = argparse.ArgumentParser(description='Train DEGAGON')
parser.add_argument('in_file',type=str, help="Input file with data structures")
args = parser.parse_args()
in_file = args.in_file

# ============================================================================================= #
# LOAD DATA
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")

# ============================================================================================= #
dropout = 0.1
nz = ppi_adj.count_nonzero()
new_ppi_adj = dropout_sparse(ppi_adj,1-dropout,nz)
print('Old number of nonzero elements: ',nz)
print('New number of nonzero elements: ',new_ppi_adj.count_nonzero())
