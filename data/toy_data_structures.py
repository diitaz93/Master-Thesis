#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# toy_data_structures.py                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 15/06/2020                                                                     #
# ============================================================================================= #
"""
Generates a random dataset with the structure of the real data used to train DECAGON. The 
parameters: number of genes, number of drugs, number of joint side effects, number of single 
side effects and number of protein features have to be set manually. The code generates 
adjacency matrices of similar density to the ones generated with real data, and enumeration 
dictionaries. Finally, it exports them in a `pickle` readable format. Thresholds of random 
matrices have been tunned heuristically.

"""
# ============================================================================================= #
import numpy as np
import scipy.sparse as sp
import pickle
from joblib import Parallel, delayed
# Parameters
n_genes = 16269
n_drugs = 630
n_se_combo = 6
n_se_mono = 9688
n_pf = 5
# ============================================================================================= #
# DATA GENERATION
# Adjacency matrix for PPI network
b = 10 * np.random.randn(n_genes, n_genes)
ppi_adj = sp.csr_matrix(((b + b.T)/2 > 20).astype(int))
ppi_degrees = np.array(ppi_adj.sum(axis=0)).squeeze()
# Adjacency matrix for DTI network
dti_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 29).astype(int))
# DDI adjacency matrices
t = n_se_combo
thresh = np.geomspace(8,20,t)
def se_adj_matrix(i):
    b = 10 * np.random.randn(n_drugs, n_drugs)
    mat = sp.csr_matrix(((b + b.T)/2 > i).astype(int))
    return mat
ddi_adj_list = Parallel(n_jobs=8)\
    (delayed(se_adj_matrix)(d) for d in thresh[:n_se_combo])
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
# Drug feature matrix
drug_feat = sp.csr_matrix((10 * np.random.randn(n_drugs, n_se_mono) > 19).astype(int))
# Protein feature matrix
prot_feat = sp.csr_matrix((10 * np.random.randn(n_genes, n_pf) > 0.5).astype(int))
# ============================================================================================= #
# CONTROL PRINTING
# Interactions (edges)
print('Interactions (edges)')
print('Number of PPI interactions:', np.sum(ppi_adj))
print('The PPI adj matrix is filled in a',round(np.sum(ppi_adj)/pow(n_genes,2)*100,2),'%')

print('Number of DTI interactions:', np.sum(dti_adj))
print('The DTI adj matrix is filled in a',round(np.sum(dti_adj)/
                                                (n_genes*n_drugs)*100,2),'%')
print('Number of DDI interactions:', np.sum(np.fromiter((np.sum(x) for x in ddi_adj_list),int)))
print('The DDI adj matrix is filled in average in a',
      round(np.mean(np.fromiter
                    ((np.sum(x)/(n_drugs*n_drugs)*100 for x in ddi_adj_list),float)),2),'%')
print('Number of DSE interactions:', np.sum(drug_feat))
print('The DSE adj matrix is filled in a',round(np.sum(drug_feat)/(n_drugs*n_se_mono)*100,2),'%')
print('The PF adj matrix is filled in a',round(np.sum(prot_feat)/(n_genes*n_pf)*100,2),'%')
print('\n')
# Drugs and genes (nodes)
print('Drugs and genes (nodes)')
print('Number of genes:', n_genes)
print('Number of drugs:', n_drugs)
print('\n')
# Side effects
print('Side effects')
print('Number of joint side effects:', n_se_combo)
print('Number of single side effects:', n_se_mono)
print('\n')
# Protein side effects
print('Number of protein features:',n_pf)
# ============================================================================================= #
# SAVING DATA STRUCTURES
data = {}
# Dictionaries
data['gene2idx'] = {i:i for i in range(n_genes)}
data['drug2idx'] = {i:i for i in range(n_drugs)}
data['se_mono_name2idx'] = {i:i for i in range(n_se_mono)}
data['se_combo_name2idx'] = {i:i for i in range(n_se_combo)}
# DDI
data['ddi_adj_list'] = ddi_adj_list
data['ddi_degrees_list'] = ddi_degrees_list
# DTI
data['dti_adj'] = dti_adj
# PPI
data['ppi_adj'] = ppi_adj
data['ppi_degrees'] = ppi_degrees
# DSE
data['drug_feat'] = sp.csr_matrix((10 * np.random.randn(n_drugs, n_se_mono) > 15).astype(int))
# PF
data['prot_feat'] = sp.csr_matrix((10 * np.random.randn(n_genes, n_pf) > 15).astype(int))
# Pickle saving
filename = './data_structures/DS/DS_toy_DSE_' + str(n_se_mono) + '_PF_'+str(n_pf) +\
'_genes_'+str(n_genes) + '_drugs_' + str(n_drugs) + '_se_' + str(n_se_combo)
with open(filename, 'wb') as f:
    pickle.dump(data, f, protocol=3)
