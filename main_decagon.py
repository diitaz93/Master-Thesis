#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# main_decagon.py                                                                               #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Runs DECAGON over a consistent real dataset with single drug side effects and protein features.

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
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

# FILE PATHS
parser = argparse.ArgumentParser(description='Train DEGAGON')
parser.add_argument('in_file',type=str, help="Input file with data structures")
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
DSE = False
if 'DSE' in words: DSE = True
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
def sigmoid(x):
        return 1. / (1 + np.exp(-x))
def accuracy(edges_pos,edges_neg,pred):
    """Gives the accuracy of the model given a set of positive and negative entries of 
    a matrix, and the probability scores of each entry. The method uses np.round to turn
    probabilities into labels 0 or 1 and feed them to the accuracy_score method of sci-kit
    learn.
    """
    pos_labels = np.ones(np.shape(edges_pos)[0])
    neg_labels = np.zeros(np.shape(edges_neg)[0])
    labels = np.hstack((pos_labels,neg_labels))
    pos_preds=[]
    scores = np.round(sigmoid(pred))
    for i,j in edges_pos:
        pos_preds.append(scores[i,j])
    neg_preds=[]
    for i,j in edges_neg:
        neg_preds.append(scores[i,j])
    predictions=np.hstack((pos_preds,neg_preds))
    return metrics.accuracy_score(labels,predictions)

def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    # Predict on set of edges
    preds = []
    #actual = []
    #predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'
        #actual.append(edge_ind)
        #predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'
        #predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    #predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]
    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    #apk_sc = rank_metrics.apk(actual, predicted, k=50)
    acc = metrics.accuracy_score(labels_all, np.round(preds_all))
    return roc_sc, aupr_sc, acc

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
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
# ============================================================================================= #
# SETTINGS AND PLACEHOLDERS
val_test_size = 0.15
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
#PRINT_PROGRESS_EVERY = 150
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
# Metric structures initialization
validation_metrics = np.zeros([num_edge_types,3,1])
train_acc = np.zeros([FLAGS.epochs,num_edge_types])
val_acc = np.zeros([FLAGS.epochs,num_edge_types])
vm_layer = np.zeros([num_edge_types,3,1])
# Start training
print("Train model")
for epoch in range(FLAGS.epochs):
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
        # Metrics
        if batch_edge_type in edge_count:
            val_auc, val_auprc, val_acs = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])
            step_time = time.time() - t
            vm_layer[batch_edge_type,:,0] = [val_auc,val_auprc,val_acs]
            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), 
                  "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc),
                  "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_acc=", "{:.5f}".format(val_acs),
                  "time=", "{:.5f}".format(step_time))
            edge_count.remove(batch_edge_type)
        itr += 1
    # Train accuracy over all train data per epoch
    for r in range(num_edge_types):
        i,j,k = minibatch.idx2edge_type[r]
        true_train_edges = minibatch.train_edges[i,j][k]
        false_train_edges = minibatch.train_edges_false[i,j][k]
        true_val_edges = minibatch.val_edges[i,j][k]
        false_val_edges = minibatch.val_edges_false[i,j][k]
        feed_dict.update({placeholders['batch_edge_type_idx']:k})
        feed_dict.update({placeholders['batch_row_edge_type']: i})
        feed_dict.update({placeholders['batch_col_edge_type']: j})
        pred = sess.run(opt.predictions,feed_dict=feed_dict)
        train_acc[epoch,r] = accuracy(true_train_edges,false_train_edges,pred)
        val_acc[epoch,r] = accuracy(true_val_edges,false_val_edges,pred)
    validation_metrics = np.concatenate((validation_metrics,vm_layer),axis=2)
    output_data['val_auc'] = validation_metrics[:,0,1:]
    output_data['val_auprc'] = validation_metrics[:,1,1:]
    output_data['train_acc'] = train_acc
    output_data['val_acc'] = val_acc
    output_data['epoch'] = epoch + 1
    
    with open(out_file,'wb') as f:
        pickle.dump(output_data, f, protocol=2)
    vm_layer = np.zeros([num_edge_types,3,1])
    
# End of training. Metric structure handling   
print("Optimization finished!")
test_scores = np.zeros([num_edge_types,3])
for et in range(num_edge_types):
    roc_score, auprc_score, acc_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test Accuracy score", "{:.5f}".format(acc_score))
    print()
    test_scores[et,0] = roc_score
    test_scores[et,1] = auprc_score
    test_scores[et,2] = acc_score
output_data['test_scores'] = test_scores
memUse = ps.memory_info()
print('Virtual memory:', memUse.vms)
print('RSS Memory:', memUse.rss)
total_time=time.time()-start
output_data['time'] = total_time
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
with open(out_file,'wb') as f:
    pickle.dump(output_data, f, protocol=2)
print("Total time:",total_time)
