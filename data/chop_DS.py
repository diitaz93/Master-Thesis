#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# chop_DS.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 22/09/2020                                                                     #
# ============================================================================================= #
"""
Takes the data structures of a network and calculates BDM of the PPI matrix. Discards a given 
fraction of the edges and updates the DS file with a reduced version of the PPI and DTI matrices.

Parameters
----------
in_file : string 
    (Relative) path to the file of data structures.
cut_frac : float, optional, flagged
    Fraction of the edges that will be discarded (between 0 and 1). Defaults to 0.25.
"""
# ============================================================================================= #
import numpy as np
import scipy.sparse as sp
import pickle
import argparse
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from decagon.utility.preprocessing import sparse_to_tuple
from pybdm import BDM
from pybdm.utils import decompose_dataset
from pybdm.partitions import PartitionRecursive
from algorithms import PerturbationExperiment, NodePerturbationExperiment

parser = argparse.ArgumentParser(description='Two possible arguments, only second one is optional.')
parser.add_argument('in_file',type=str, help="Input DS file.")
parser.add_argument('--cut_frac', type=float, default=0.25,\
                    help="Fraction of edges to be discarded")
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
sim_type = words[2]
# Fraction of edges to be discarded
cut_frac = args.cut_frac

# DECAGON sparse matrix function
#def sparse_to_tuple(sparse_mx):
#    if not sp.isspmatrix_coo(sparse_mx):
#        sparse_mx = sparse_mx.tocoo()
#    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#    values = sparse_mx.data
#    shape = sparse_mx.shape
#    return coords, values, shape

# Import original Data structures
print('\n==== IMPORTED VARIABLES ====')
with open(in_file,'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
old_genes = len(gene2idx)
print('\n')
# =================================== BDM =============================================== #
ppi_mat = ppi_adj.todense() # High memory requirement for big matrices
# Calculate algorithmic complexity
bdm = BDM(ndim=2, partition=PartitionRecursive)
ppi_per = PerturbationExperiment(bdm,metric='bdm',bipartite_network=False)
ppi_per.set_data(np.array(ppi_mat))
edge_complexity = ppi_per.run()
# Reshape to the adj matrix shape
complexity_mat = edge_complexity.reshape(np.shape(ppi_adj))
#============================= PRELIMINARY SAVING OF BDM ================================ #
out_file_bdm = 'data_structures/BDM/EDGES_PPI_'+sim_type+'_genes_' + str(old_genes)
print('Output BDM file: ',out_file_bdm,'\n')
with open(out_file_bdm,'wb') as f:
    pickle.dump(edge_complexity, f)
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
print('==== CHANGES MADE ====')
print('Nonzero entries before',len(coords))
print('Nonzero entries after',new_ppi_adj.count_nonzero())
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
    print('New shape of DTI matrix: ',np.shape(new_dti_adj))
else:
    print('No further modifications to the matrices are needed')
new_ppi_adj = sp.csr_matrix(new_ppi_adj)
new_dti_adj = sp.csr_matrix(new_dti_adj)
# ================================= EXPORT AND SAVING ========================================= #
n_genes = len(gene2idx)
n_drugs = len(drug2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
print('Previous number of genes: ',old_genes)
print('New number of genes: ',n_genes)
print('\n')

# Dictionaries
data = {}
data['gene2idx'] = gene2idx
data['drug2idx'] = drug2idx
data['se_mono_name2idx'] = se_mono_name2idx
data['se_combo_name2idx'] = se_combo_name2idx
# DDI
data['ddi_adj_list'] = ddi_adj_list
data['ddi_degrees_list'] = ddi_degrees_list
# DTI
data['dti_adj'] = new_dti_adj
# PPI
data['ppi_adj'] = new_ppi_adj
data['ppi_degrees'] = new_ppi_degrees
# DSE
data['drug_feat'] = drug_feat

# SAVING
out_file = 'data_structures/CHOP/DS_' + sim_type + '_cutfrac_'+str(cut_frac) +\
        '_DSE_' + str(n_se_mono) + '_genes_' +str(n_genes) + '_drugs_' + str(n_drugs) +\
        '_se_' + str(n_se_combo)
print('Output file: ',out_file,'\n')
with open(out_file,'wb') as f:
    pickle.dump(data, f)

