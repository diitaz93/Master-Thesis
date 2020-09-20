#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# new_adj_ppi.py                                                                                #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 16/09/2020                                                                     #
# ============================================================================================= #
"""
Takes the adjacency matrix of a protein-protein interaction (PPI) network and uses algorithmic
complexity to discard the least relevant interactions (edges) of the network, saving a new 
adjacency matrix in the output file.

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
# Fraction of edges to be discarded
cut_frac = 0.25

#Define function for sparse matrices of DECAGON
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Import original PPI matrix
with open(in_file,'rb') as f:
    ppi_adj = pickle.load(f)['ppi_adj']

# Calculate algorithmic complexity
bdm = BDM(ndim=2, partition=PartitionRecursive)
ppi_per = PerturbationExperiment(bdm,metric='bdm',bipartite_network=False)
ppi_per.set_data(np.array(ppi_adj.todense()))
edge_complexity = ppi_per.run()
# Reshape to the adj matrix shape
complexity_mat = edge_complexity.reshape(np.shape(ppi_adj))

# Get coordinates and complexities of positive edges
coords,_,_ = sparse_to_tuple(ppi_adj)
l= np.shape(coords)[0]
true_cmplx = np.abs(complexity_mat[coords[:,0],coords[:,1]].reshape(615,1))
# Use dummy column to keep track of indices
a = np.concatenate((np.abs(true_cmplx),np.arange(615).reshape(615,1)),axis=1)
sorted_values = a[a[:,0].argsort()[::-1]]
# Discard the lowest complexity edges
remain = np.arange(np.floor(l*(1-cut_frac)),dtype=int)
new_values = sorted_values[remain,:]
indices = new_values[:,1].astype(int)
new_coords = coords[indices,:]
new_l = np.shape(new_coords)[0]
# New adjacency matrix
new_ppi_adj = sp.csr_matrix((np.ones(new_l), (new_coords[:,0], new_coords[:,1])), shape=np.shape(ppi_adj))
# SAVING
out_file = 'ADJ_PPI_cutfrac_'+str(cut_frac)
print(out_file)
#with open(out_file,'wb') as f:
#    pickle.dump(new_ppi_adj, f)
    
