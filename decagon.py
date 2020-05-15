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
import time
import os
import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
import pandas as pd
import psutil
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
from data.load_functions import *

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
# Loading Gene data (PPI)
ppi, gene2idx = load_ppi(fname='data/clean_data/ppi_mini.csv')
ppi_adj = nx.adjacency_matrix(ppi)
ppi_degrees = np.array(ppi_adj.sum(axis=0)).squeeze() 
ppi_genes = ppi.number_of_nodes() # Number of genes (nodes)
# Loading individual side effects
stitch2se, semono2name, semono2idx = load_mono_se(fname='data/clean_data/mono_mini.csv')
n_semono = len(semono2name)
print('Number of individual side effects: ', n_semono)
# Loading Target data (DTI)
stitch2proteins = load_targets(fname='data/clean_data/targets_mini.csv')
dti_drugs = len(pd.unique(stitch2proteins.keys()))
dti_genes = len(set(chain.from_iterable(stitch2proteins.itervalues())))
print('Number of genes in DTI:', dti_genes)
print('Number of drugs in DTI:', dti_drugs)
# Loading Drug data (DDI)
combo2stitch, combo2se, secombo2name, drug2idx = load_combo_se(fname='data/clean_data/combo_mini.csv')
# Loading Side effect data (features)
stitch2se, semono2name, semono2idx = load_mono_se(fname='data/clean_data/mono_mini.csv')
# Loading protein features
PF = pd.read_csv('data/clean_data/genes_mini.csv', sep=',',header=None).to_numpy()
ddi_drugs = len(drug2idx)
print('Number of drugs: ', ddi_drugs)
# ============================================================================================= #
# PREPROCESS DATA
# Drug-target adjacency matrix
dti_adj = np.zeros([ppi_genes,ddi_drugs],dtype=int)
for drug in drug2idx.keys():
    for gene in stitch2proteins[drug]:
        if gene==set():
            continue
        else:
            idp = gene2idx[str(gene)]
            idd = drug2idx[drug]
            dti_adj[idp,idd] = 1  
dti_adj = sp.csr_matrix(dti_adj)
# DDI adjacency matrix
ddi_adj_list = []
for se in secombo2name.keys():
    m = np.zeros([ddi_drugs,ddi_drugs],dtype=int)
    for pair in combo2se.keys():
        if se in combo2se[pair]:
            d1,d2 = combo2stitch[pair]
            m[drug2idx[d1],drug2idx[d2]] = m[drug2idx[d2],drug2idx[d1]] = 1
    ddi_adj_list.append(sp.csr_matrix(m))    
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]

adj_mats_orig = {
    (0, 0): [ppi_adj, ppi_adj.transpose(copy=True)],
    (0, 1): [dti_adj],
    (1, 0): [dti_adj.transpose(copy=True)],
    (1, 1): ddi_adj_list + [x.transpose(copy=True) for x in ddi_adj_list],
}
degrees = {
    0: [ppi_degrees, ppi_degrees],
    1: ddi_degrees_list + ddi_degrees_list, 
}
# featureless (genes)
#gene_feat = sp.identity(n_genes)
gene_feat = sp.coo_matrix(PF)
gene_nonzero_feat, gene_num_feat = 2*[gene_feat.shape[1]]
gene_feat = preprocessing.sparse_to_tuple(gene_feat)#.tocoo())
# features (drugs)
oh_feat = np.zeros([ddi_drugs,n_semono], dtype=int)
for drug in drug2idx.keys():
    for se in stitch2se[drug]:
        did = drug2idx[drug]
        seid = semono2idx[se]
        oh_feat[did,seid] = 1
drug_feat = sp.csr_matrix(oh_feat)
drug_nonzero_feat = n_semono
drug_num_feat = n_semono
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())
# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}
# Dictionary with the shape of all the matrices of the dictionary adj_mats_orig
edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}
#Dictionary with the number of matrices for each entry of adj_mats_orig
edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)
# ============================================================================================= #
# SETTINGS AND PLACEHOLDERS
val_test_size = 0.05
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
f = open('training.txt', 'wb')
acc_scores = np.array([None,None,None,None,None])
print("Train model")
f.write("Train model\n")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
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

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])
             step_time = time.time() - t

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(step_time))
            f.write("Epoch:%04d Iter:%04d Edge:%04d train_loss=%.5f val_roc=%.5f val_auprc=%.5f val_apk=%.5f time=%.5f\n"% 
                 ((epoch + 1),(itr + 1), batch_edge_type, train_cost, val_auc, val_auprc, val_apk, step_time))
            acc_scores = np.vstack((acc_scores,[val_auc,val_auprc,val_apk,train_cost,step_time]))
        itr += 1

acc_scores=acc_scores[1:,:]
np.save('./acc_scores.npy',acc_scores)
print("Optimization finished!")
f.write("Optimization finished!\n")
for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    f.write("Edge type=[%02d, %02d, %02d]\n" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    f.write("Edge type:%04d Test AUROC score %.5f\n" % (et, roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    f.write("Edge type:%04d Test AUROC score %.5f\n" % (et, auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    f.write("Edge type:%04d Test AUROC score %.5f\n" % (et, apk_score))
    print()
memUse = ps.memory_info()
print('Virtual memory:', memUse.vms)
f.write('Virtual memory:%f\n'% memUse.vms)
print('RSS Memory:', memUse.rss)
f.write('RSS Memory:%f\n'% memUse.rss)
total_time=time.time()-start
np.save('./mem_and_time.npy',np.array([memUse.vms,memUse.rss,total_time]))
print("Total time:",total_time)
f.write("Total time:%f\n" % total_time)
f.close()
