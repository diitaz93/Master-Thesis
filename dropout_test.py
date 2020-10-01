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
words = in_file.split('_')
d_text = ''
# ============================================================================================= #
# LOAD DATA
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")

mb_file = 'data/data_structures/MINIBATCH/MINIBATCH_real_genes_19081_drugs_639_se_964_batchsize_512_valsize_0.15'
with open(mb_file, 'rb') as f:
    minibatch = pickle.load(f)

def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'neg_batch': tf.placeholder(tf.int32, name='neg_batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

# ============================================================================================= #
dropout = 0.1

feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
feed_dict = minibatch.update_feed_dict(feed_dict=feed_dict,
            dropout=dropout,
            placeholders=placeholders)
mat =  placeholders['feat_0']
nz = nonzero_feat[0]
new_mat = dropout_sparse(mat,1-dropout,nz)
sess = tf.Session()
print('Old number of nonzero elements: ',nz)
print('New number of nonzero elements: ',sess.run(new_mat))
