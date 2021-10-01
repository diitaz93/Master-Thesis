#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# noise_minibatch.py                                                                            #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 28/07/2020                                                                     #
# ============================================================================================= #
"""
Loads a minibatch file and modifies the structures test_edges and test_edges_false, interchanging
a fraction of edges between them. The fraction is given by the variable noise, and it is a number
between 0 and 1.

Parameters
----------
in_file : string
    (Relative) path to the minibatch file.
"""
# ============================================================================================= #
from __future__ import print_function
import argparse
import os
import numpy as np
import pickle
from decagon.deep.minibatch import EdgeMinibatchIterator

# FILE PATHS AND IMPORT MINIBATCH
parser = argparse.ArgumentParser(description='MINIBATCH file')
parser.add_argument('in_file',type=str, help="Input file with data structures")
args = parser.parse_args()
mb_file = args.in_file
with open(mb_file, 'rb') as f:
    minibatch = pickle.load(f)
# NOISE VALUE
noise = 0.05
# ============================================================================================= #
# MODIFICATION OF TEST DATASET IN MINIBATCH
for edge in minibatch.test_edges:
    for p in range(np.shape(minibatch.test_edges[edge])[0]):
        data_true = minibatch.test_edges[edge][p]
        data_false = minibatch.test_edges_false[edge][p]
        l = len(data_true)
        assert l==len(data_false), 'Dimension of true and false mismatch'
        full_idx = np.arange(2*l)
        n_idx = np.floor(noise*2*l).astype(np.int)
        idx = np.random.choice(full_idx,size=n_idx)
        col = np.zeros(len(idx))
        # We choose a convention that indices from l to 2l are from the true dataset
        for i in range(len(idx)):
            if idx[i] >= l:
                idx[i] = idx[i]-l
                col[i] = 1
        col = col.astype(np.bool)
        false_to_true = idx[~col]
        true_to_false = idx[col]
        data_true = np.concatenate((data_true,data_false[false_to_true]),axis=0)
        data_false = np.concatenate((data_false,data_true[true_to_false]),axis=0)
        minibatch.test_edges[edge][p] = np.delete(data_true, true_to_false, axis=0)
        minibatch.test_edges_false[edge][p] = np.delete(data_false, false_to_true, axis=0)
print('Noise added to the test dataset')
# ============================================================================================= #
# MINIBATCH SAVING
out_file = mb_file + '_noise_' + str(noise)
with open(out_file,'wb') as f:
    pickle.dump(minibatch, f, protocol=2)
print('Modified minibatch saved')
