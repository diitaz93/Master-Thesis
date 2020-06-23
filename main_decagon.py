#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# test.py (currently developing)                                                                #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Runs DECAGON over a consistent real dataset with single drug side effects and protein features. 
"""
# ============================================================================================= #
from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations, chain
import argparse
import time
import os
import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
import pandas as pd
import psutil
import pickle
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

parser = argparse.ArgumentParser(description='Train DEGAGON')
parser.add_argument('in_file',type=str, help="Input file with data structures")
parser.add_argument('out_file',type=str,help="Output file root with TRAIN info")
args = parser.parse_args()
in_file = args.in_file
out_file = args.out_file

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# Train on GPU
#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

# BEGIN
start = time.time()
pid = os.getpid()
ps= psutil.Process(pid)

# ============================================================================================= #
# FUNCTIONS

def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc

def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
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
# ============================================================================================= #
# SETTINGS AND PLACEHOLDERS
val_test_size = 0.05
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150
print("Defining placeholders")
placeholders = construct_placeholders(edge_types)
# ============================================================================================= #
# CREATE MINIBATCH ITERATOR, MODEL AND OPTIMIZER
print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)
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
output_data = {}
acc_scores = np.zeros([num_edge_types,4,1])
print("Train model")
for epoch in range(FLAGS.epochs):
    acc_layer = np.zeros([num_edge_types,4,1])
    minibatch.shuffle()
    itr = 0
    edge_count = range(num_edge_types)
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        #if itr % PRINT_PROGRESS_EVERY == 0:
        if batch_edge_type in edge_count:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])
            step_time = time.time() - t
            acc_layer[batch_edge_type,:,0] = [val_auc,val_auprc,val_apk,train_cost]
            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:",
                  "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(step_time))
            edge_count.remove(batch_edge_type)

        itr += 1
    acc_scores = np.concatenate((acc_scores,acc_layer),axis=2)
    output_data['val_auc'] = acc_scores[:,0,1:]
    output_data['val_auprc'] = acc_scores[:,1,1:]
    output_data['val_apk'] = acc_scores[:,2,1:]
    output_data['train_cost'] = acc_scores[:,3,1:]
    output_data['epoch'] = epoch + 1
    with open(filename, 'wb') as f:
        pickle.dump(output_data, f, protocol=2)
    acc_layer = np.zeros([num_edge_types,5,1])
        
print("Optimization finished!")
final_scores = np.zeros([num_edge_types,4,1])
for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()
    final_scores[et,0,0] = roc_score
    final_scores[et,1,0] = auprc_score
    final_scores[et,2,0] = apk_score
acc_scores = np.concatenate((acc_scores,final_scores),axis=2)
output_data['val_auc'] = acc_scores[:,0,1:]
output_data['val_auprc'] = acc_scores[:,1,1:]
output_data['val_apk'] = acc_scores[:,2,1:]
output_data['train_cost'] = acc_scores[:,3,1:]
memUse = ps.memory_info()
print('Virtual memory:', memUse.vms)
print('RSS Memory:', memUse.rss)
total_time=time.time()-start
output_data['time'] = total_time
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
print("Total time:",total_time)
filename = 'results_training/'+out_file+'_epochs'+str(FLAGS.epochs)+'_h1'+\
           str(FLAGS.hidden1)+'_h2'+str(FLAGS.hidden2)+'_lr'+str(FLAGS.learning_rate)+\
           '_dropout'+str(FLAGS.dropout)
with open(filename, 'wb') as f:
    pickle.dump(output_data, f, protocol=2)
