#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# from chop_DS.py                                                                               #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 22/09/2020                                                                     #
# ============================================================================================= #
"""
Fix!

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
"""
# ============================================================================================= #
import numpy as np
import scipy.sparse as sp
import pickle
from pybdm import BDM
from pybdm.utils import decompose_dataset
from pybdm.partitions import PartitionIgnore
from pybdm.partitions import PartitionRecursive
from algorithms import PerturbationExperiment, NodePerturbationExperiment
import argparse

parser = argparse.ArgumentParser(description='DS file')
parser.add_argument('in_file',type=str, help="Input file with data structures")
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
sim_type = words[2]
# Fraction of edges to be discarded
cut_frac = 0.01
# DECAGON sparse matrix function
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Import original Data structures
with open(in_file,'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
old_genes = len(gene2idx)
old_drugs = len(drug2idx)
old_se_combo = len(se_combo_name2idx)
old_se_mono = len(se_mono_name2idx)
# =================================== BDM =============================================== #
# Load algorithmic complexity vector
out_file_bdm = 'data_structures/BDM/EDGES_PPI_real_genes_19081'
print(out_file_bdm)
with open(out_file_bdm,'rb') as f:
    edge_complexity = pickle.load(f)
complexity_mat = edge_complexity.reshape(np.shape(ppi_adj))
print('Is it symmetric?',np.array_equal(complexity_mat,complexity_mat.T))
# =============================== REMOVING EDGES ======================================== #
coords,_,_ = sparse_to_tuple(ppi_adj)
# Take the upper triangular coordinates
upper_coords = coords[(coords[:,1]-coords[:,0]>0).nonzero()]
# Select abs of the complexity of selected entries
true_cmplx = np.abs(complexity_mat[upper_coords[:,0],upper_coords[:,1]]).squeeze()
# Give an index to the edge
pair = np.array(list(enumerate(true_cmplx)))
# Sort from greatest to lowest complexity
sorted_pair = pair[pair[:,1].argsort()][::-1]
# Select sorted indices
idx = sorted_pair[:,0].astype(int)
# Select a threshold entry according to the cut fraction
threshold = np.floor(len(idx)*(1-cut_frac)).astype(int)
# Select indices above threshold
idx = idx[:threshold]
# Generate row and col indices of full matrix
row_ind = np.concatenate((upper_coords[idx,0],upper_coords[idx,1]),axis=0)
col_ind = np.concatenate((upper_coords[idx,1],upper_coords[idx,0]),axis=0)
# Form the new adjacency matrix
new_ppi_adj = sp.csr_matrix((np.ones(2*threshold), (row_ind, col_ind)),\
                            shape=np.shape(ppi_adj),dtype=int)
print('Nonzero entries before',len(coords))
print('Nonzero entries after',new_ppi_adj.count_nonzero())
print('Is it symmetric?',np.array_equal(new_ppi_adj.todense(),new_ppi_adj.todense().T))
# ==================== NETWORK CONSISTENCY ============================================== # 
# Find rows of zeros (indices)
new_ppi_adj = new_ppi_adj.todense()
genes_zero = np.asarray(~new_ppi_adj.any(axis=1)).nonzero()[0]
print('Number of zero rows/columns in PPI matrix: ',len(genes_zero))
# If there are
if len(genes_zero)>0:
    #### PPI ####
    # Delete those rows and columns
    new_ppi_adj = np.delete(np.delete(new_ppi_adj,genes_zero,axis=1),genes_zero,axis=0)
    print('New shape PPI matrix: ',np.shape(new_ppi_adj))
    # Update index dictionary
    gene_dict = {key:val for key, val in gene2idx.items() if val not in genes_zero}
    gene2idx = {gene:i for i, gene in enumerate(gene_dict.keys())}
    # Update degree list
    new_ppi_degrees = np.array(new_ppi_adj.sum(axis=0).astype(int)).squeeze()
    #### DTI ####
    # Deletes the corresponding rows in DTI
    new_dti_adj = dti_adj.todense()
    new_dti_adj = np.delete(new_dti_adj,genes_zero,axis=0)
    #### DRUGS ####
    # Finds drugs that became disconnected from network (indices)
    drugs_zero = np.asarray(~new_dti_adj.any(axis=0)).squeeze().nonzero()[0]
    print('Number of disconnected drugs: ',len(drugs_zero))
    # Remove drugs from DTI matrix
    new_dti_adj = np.delete(new_dti_adj,drugs_zero,axis=1)
    # Remove drugs from drug feature matrix
    new_drug_feat = drug_feat.todense()
    new_drug_feat = np.delete(new_drug_feat,drugs_zero,axis=0)
    # Find drug side effects that have no drug
    mono_zero = np.asarray(~new_drug_feat.any(axis=0)).squeeze().nonzero()[0]
    print('Number of side effects without drug: ',len(mono_zero))
    if len(mono_zero)>0:
        # Remove them from drug feature matrix
        new_drug_feat = np.delete(new_drug_feat,mono_zero,axis=1)
        # Update index dictionary
        mono_dict = {key:val for key,val in se_mono_name2idx.items() if val not in mono_zero}
        se_mono_name2idx = {se: i for i, se in enumerate(mono_dict.keys())}
    print('New shape of drug features matrix: ',np.shape(new_drug_feat))
    #### DDI ####
    # Remove drugs from adjacency matrices
    new_ddi_degrees_list = []
    new_ddi_adj_list = []
    for i in ddi_adj_list:
        # Remove drugs from DDI matrices
        ddi_mat = np.delete(np.delete(i.todense(),drugs_zero,axis=0),\
                                    drugs_zero,axis=1)
        new_ddi_adj_list.append(sp.csr_matrix(ddi_mat))
        # Update degree list
        new_ddi_degrees_list.append(np.array(ddi_mat.sum(axis=0)).squeeze())
    # Update index dictionary
    drug_dict = {key:val for key, val in drug2idx.items() if val not in drugs_zero}
    drug2idx = {drug: i for i, drug in enumerate(drug_dict.keys())}
    print('New shape of DTI matrix: ',np.shape(new_dti_adj))
    print('New shape of DDI matrices: ',np.shape(new_ddi_adj_list[0]))
else:
    print('No further modifications to the matrices are needed')
new_drug_feat = sp.csr_matrix(new_drug_feat)
new_ppi_adj = sp.csr_matrix(new_ppi_adj)
new_dti_adj = sp.csr_matrix(new_dti_adj)
# ================================= EXPORT AND SAVING ========================================= #
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
print('Previous number of genes: ',old_genes)
print('New number of genes: ',n_genes)
print('Previous number of drugs: ',old_drugs)
print('New number of drugs: ',n_drugs)
print('Previous number of joint side effects: ',old_se_combo)
print('New number of joint side effects: ',n_se_combo)
print('Previous number of single side effects: ',old_se_mono)
print('New number of single sige effects: ',n_se_mono)

# Dictionaries
data = {}
data['gene2idx'] = gene2idx
data['drug2idx'] = drug2idx
data['se_mono_name2idx'] = se_mono_name2idx
data['se_combo_name2idx'] = se_combo_name2idx
# DDI
data['ddi_adj_list'] = new_ddi_adj_list
data['ddi_degrees_list'] = new_ddi_degrees_list
# DTI
data['dti_adj'] = new_dti_adj
# PPI
data['ppi_adj'] = new_ppi_adj
data['ppi_degrees'] = new_ppi_degrees
# DSE
data['drug_feat'] = new_drug_feat

# SAVING
out_file = 'data_structures/CHOP/DS_' + sim_type + '_cutfrac_'+str(cut_frac) +\
        '_DSE_' + str(n_se_mono) + '_genes_' +str(n_genes) + '_drugs_' + str(n_drugs) +\
        '_se_' + str(n_se_combo)
print(out_file)
with open(out_file,'wb') as f:
    pickle.dump(data, f)

