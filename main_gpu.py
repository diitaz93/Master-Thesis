#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# main_gpu.py                                                                                   #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Trains and tests DECAGON over a consistent dataset exporting performance in a pickle python2 
readable file. 
It recieves as parameter the path of a pickle file containing the specifications and the data 
structures. This file defines which features are going to be used, that can be None, Single drug 
side effects, Algortithmic complexity of the subnetworks or both.
This scripts runs the code on GPU.

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
gpu : int, default=0
    ID of desired GPU.
--epochs : int (optional), defaults to 50
    Number of epochs (how many times the whole dataset is used to train the model).
-- batch_size : int (optional), defaults to 512.
    Number of training instances used per batch in SGD.
"""
# ============================================================================================= #
from __future__ import division
from __future__ import print_function
import argparse
import time
import datetime
import os
import warnings
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
import pandas as pd
import psutil
import pickle
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# FILE PATHS
parser = argparse.ArgumentParser(description='Train DEGAGON')
parser.add_argument('in_file',type=str, help="Path of file with data structures")
parser.add_argument('gpu', nargs='?',default =0,type=int, help="GPU_ID")
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=512, help='Size of batch')
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
DSE = False
BDM = False
if 'DSE' in words: DSE = True
if 'BDM' in words: BDM = True
# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# BEGIN
start = time.time()
pid = os.getpid()
ps= psutil.Process(pid)
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)
# ============================================================================================= #
# FUNCTIONS
def sigmoid(x):
        return 1. / (1 + np.exp(-x))

def get_accuracy_scores(edges_pos, edges_neg, edge_type, noise=False):
    """ Returns the AUROC, AUPRC and Accuracy of the dataset corresponding to the edge
    'edge_type' given as a tuple. The parameters 'edges_pos' and 'edges_neg' are the list 
    of edges of positive and negative interactions respectively of a given dataset, i.e., 
    train, test or validation.
    """
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)
    # Predict on set of edges
    preds = []
    for u, v in edges_pos:
        score = sigmoid(rec[u, v])
        preds.append(score)
        if not noise:
                assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] > 0, 'Problem 1'
    preds_neg = []
    for u, v in edges_neg:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        if not noise:
                assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'
    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    acc = metrics.accuracy_score(labels_all, np.round(preds_all))

    return roc_sc, aupr_sc, acc

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

# ============================================================================================= #
# LOAD DATA
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
# ============================================================================================= #
# SETTINGS AND PLACEHOLDERS
val_test_size = 0.15
noise = 0
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', args.epochs, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', args.batch_size, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
print("Defining placeholders")
placeholders = construct_placeholders(edge_types)
# ============================================================================================= #
# LOAD MINIBATCH ITERATOR, AND CREATE MODEL AND OPTIMIZER
noise_str = bool(noise)*('_noise_' + str(noise))
print("Load minibatch iterator")
mb_file = 'data/data_structures/MINIBATCH/MINIBATCH_'+words[2]+'_genes_'+str(n_genes)+\
          '_drugs_'+str(n_drugs)+'_se_'+str(n_se_combo)+'_batchsize_'+str(FLAGS.batch_size)+\
          '_valsize_'+str(val_test_size) + noise_str

with open(mb_file, 'rb') as f:
    MB = pickle.load(f)
    for key in MB.keys():
        globals()[key]=MB[key]
        print(key,"Imported successfully")
minibatch.feat = feat
print("New features loaded to minibatch")

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)
print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )
print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}
# ============================================================================================= #
# TRAINING
# Metric structures initialization
output_data={}
out_file = 'results_training/TRAIN_'+words[2]+DSE*('_DSE_'+str(n_se_mono))+BDM*('_BDM')\
            +'_genes_'+str(n_genes)+'_drugs_'+str(n_drugs)+'_se_'+str(n_se_combo)+'_epochs_'+\
            str(FLAGS.epochs)+'_dropout_'+str(FLAGS.dropout)+'_valsize_'+\
            str(val_test_size) + noise_str
val_metrics = np.zeros([FLAGS.epochs,num_edge_types,3])
train_metrics = np.zeros([FLAGS.epochs,num_edge_types,3])
# Start training
print("Train model")
for epoch in range(FLAGS.epochs):
    minibatch.shuffle()
    t = time.time()
    itr = 0
    while not minibatch.end():
    	# Construct feed dictionary     
	feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)     
	feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)
        # Training step: run single weight update
        outs = sess.run([opt.opt_op], feed_dict=feed_dict)
        if (itr+1)%1000==0:print('Iteration',itr,' of epoch',epoch)
        itr += 1
    # Train & validation accuracy over all train data per epoch
    print('===============================================================================')
    print("Epoch", "%04d" % (epoch + 1),'finished!')
    print("Time=", "{:.5f}".format(time.time()-t))
    for r in range(num_edge_types):
        i,j,k = minibatch.idx2edge_type[r]
        print('Metrics for ', edge2name[i,j][k])
        train_metrics[epoch,r,:] = get_accuracy_scores(
            minibatch.train_edges[i,j][k], minibatch.train_edges_false[i,j][k],(i,j,k))
        val_metrics[epoch,r,:] = get_accuracy_scores(
            minibatch.val_edges[i,j][k], minibatch.val_edges_false[i,j][k],(i,j,k))
        print("AUROC:Train=", "{:.4f}".format(train_metrics[epoch,r,0])
              ,"Validation=", "{:.4f}".format(val_metrics[epoch,r,0])
              ,"AUPRC:Train=", "{:.4f}".format(train_metrics[epoch,r,1])
              ,"Validation=", "{:.4f}".format(val_metrics[epoch,r,1])
              ,"Accuracy:Train=", "{:.4f}".format(train_metrics[epoch,r,2])
              ,"Validation=", "{:.4f}".format(val_metrics[epoch,r,2]))
    output_data['val_metrics'] = val_metrics
    output_data['train_metrics'] = train_metrics
    output_data['epoch'] = epoch + 1
    with open(out_file,'wb') as f:
        pickle.dump(output_data, f, protocol=2)
    
# End of training. Metric structure handling   
print("Optimization finished!")
test_metrics = np.zeros([num_edge_types,3])
for et in range(num_edge_types):
    i,j,k = minibatch.idx2edge_type[et]
    test_metrics[et,:] = get_accuracy_scores(
            minibatch.test_edges[i,j][k], minibatch.test_edges_false[i,j][k], (i,j,k),
            noise=bool(noise))
    print("Edge type=", edge2name[i,j][k])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(test_metrics[et,0]))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(test_metrics[et,1]))
    print("Edge type:", "%04d" % et, "Test Accuracy score", "{:.5f}".format(test_metrics[et,2]))
    print()
output_data['test_metrics'] = test_metrics
memUse = ps.memory_info()
print('Virtual memory:', memUse.vms*1e-09,'Gb')
print('RSS Memory:', memUse.rss*1e-09,'Gb')
train_time=time.time()-start
output_data['train_time'] = train_time
output_data['edge2name'] = edge2name
output_data['drug2idx'] = drug2idx
output_data['gene2idx'] = gene2idx
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
with open(out_file,'wb') as f:
    pickle.dump(output_data, f, protocol=2)
print('Total time:', datetime.timedelta(seconds=train_time))
