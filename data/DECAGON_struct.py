#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# DECAGON_struct.py                                                                             #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/10/2020                                                                     #
# ============================================================================================= #
"""
Imports the matrices contained in the 'DS' file whose path is added as parameter, generates the 
data structures to feed the DECAGON model and exports them in a pickle python2 readable format.

Parameters
----------
in_file : string
    (Relative) path to the file of data structures.
--dse : int, bool or string(optional)
    If this flag is added as argument, the generated data structures will have as drug feature
    matrix the one-hot encoded vectors of single side effects. If used with the --bdm flag, the 
    drug vector will be a concatenation of both features. If no other flag is added, the 
    feature matrix will be the identity matrix. The argument can be anything as only its 
    existence would be evaluated.
--bdm : int, bool or string(optional)
    If this flag is added as argument, the generated data structures will use the algorithmic 
    complexity of the nodes as features. If used with the --dse flag, the drug vector will be a 
    concatenation of both features. If no other flag is added, the feature matrix will be the 
    identity matrix. The argument can be anything as only its existence would be evaluated.
"""
# ============================================================================================= #
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from decagon.utility.preprocessing import sparse_to_tuple

parser = argparse.ArgumentParser(description='Remove outliers from datasets')
parser.add_argument('in_file',type=str, help="Input DS file")
parser.add_argument('--dse', help='Use single side effects as drug features')
parser.add_argument('--bdm', help='Use algorithmic complexity as node features')
args = parser.parse_args()
in_file = args.in_file
words = in_file.split('_')
DSE = False
BDM = False
if args.dse:
    DSE = True
if args.bdm:
    BDM = True
# ============================================================================================= #
# IMPORT FILE
print('\n==== IMPORTED DS VARIABLES ====')
with open(in_file, 'rb') as f:
    DS = pickle.load(f)
    for key in DS.keys():
        globals()[key]=DS[key]
        print(key,"Imported successfully")
print('\n')
n_drugs = len(drug2idx)
n_genes = len(gene2idx)
n_se_combo = len(se_combo_name2idx)
n_se_mono = len(se_mono_name2idx)
# ============================================================================================= #
# IMPORT BDM
if BDM:
    # PPI
    PPI_file = './data_structures/BDM/PPI_BINBDM_real_genes_' + str(n_genes)
    print('==== IMPORTED PPI VARIABLES ====')
    with open(PPI_file, 'rb') as f:
        PPI = pickle.load(f)
        for key in PPI.keys():
            globals()[key]=PPI[key]
            print(key,"Imported successfully")
    print('\n')
    to_add_bdm_ppi = np.hstack([nodebdm_ppi.reshape(-1,1),rem_edgebdm_ppi.reshape(-1,1)])
    print('PPI feature vector shape: ',np.shape(to_add_bdm_ppi),'\n')
    
    # DTI
    DTI_file = './data_structures/BDM/DTI_BINBDM_real_genes_' + str(n_genes) + '_drugs_' +\
               str(n_drugs)
    print('==== IMPORTED DTI VARIABLES ====')
    with open(DTI_file, 'rb') as f:
        DTI = pickle.load(f)
        for key in DTI.keys():
            globals()[key]=DTI[key]
            print(key,"Imported successfully")
    print('\n')
    to_add_bdm_drugs_dti = np.hstack([nodebdm_drugs_dti.reshape(-1,1),
                                      rem_edgebdm_drugs_dti.reshape(-1,1)])
    to_add_bdm_genes_dti = np.hstack([nodebdm_genes_dti.reshape(-1,1),
                                      rem_edgebdm_genes_dti.reshape(-1,1)])
    print('DTI gene feature vector shape: ',np.shape(to_add_bdm_genes_dti))
    print('DTI drug feature vector shape: ',np.shape(to_add_bdm_drugs_dti),'\n')
    
    # DDI
    DDI_file = './data_structures/BDM/DDI_BINBDM_real_se_' + str(n_se_combo)  + '_drugs_' +\
               str(n_drugs)
    print('==== IMPORTED DDI VARIABLES ====')
    with open(DDI_file, 'rb') as f:
        DDI = pickle.load(f)
        for key in DDI.keys():
            globals()[key]=DDI[key]
            print(key,"Imported successfully")
    print('\n')
    node_ddi = np.hstack([i.reshape(-1,1) for i in nodebdm_ddi_list])
    rem_edge_ddi = np.hstack([i.reshape(-1,1) for i in rem_edgebdm_ddi_list])
    to_add_bdm_ddi = np.hstack([node_ddi,rem_edge_ddi])
    print('DDI feature vector shape: ', np.shape(to_add_bdm_ddi))
# ============================================================================================= #
# FEATURE MATRICES
prot_feat = sp.identity(n_genes)
if not DSE:
    drug_feat = sp.identity(n_drugs)
    
if BDM:
    prot_feat = np.hstack([to_add_bdm_genes_dti,to_add_bdm_ppi])
    # Drug features
    if DSE:
        drug_feat = np.asarray(np.hstack([drug_feat.todense(),
                                          to_add_bdm_drugs_dti,to_add_bdm_ddi]))
    else:
        drug_feat = np.hstack([to_add_bdm_drugs_dti,to_add_bdm_ddi])
print('Drug feature matrix shape: ',np.shape(drug_feat))
print('Protein feature matrix shape: ',np.shape(prot_feat))


# Drug features
drug_num_feat = drug_feat.shape[1]
drug_nonzero_feat = len(np.nonzero(drug_feat)[0])
drug_feat = sparse_to_tuple(sp.coo_matrix(drug_feat))
# Protein features
gene_num_feat = prot_feat.shape[1]
gene_nonzero_feat = len(np.nonzero(prot_feat)[0])
gene_feat = sparse_to_tuple(sp.coo_matrix(prot_feat))
# ============================================================================================= #
# CREATION OF DECAGON DICTIONARIES
adj_mats_orig = {
    (0, 0): [ppi_adj],
    (0, 1): [dti_adj],
    (1, 0): [dti_adj.transpose(copy=True)],
    (1, 1): ddi_adj_list,
}
degrees = {
    0: [ppi_degrees],
    1: ddi_degrees_list 
}
edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}
edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(list(edge_types.values()))
print("Edge types:", "%d" % num_edge_types)
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
edge2name = {
    (0, 0): ['PPI'],
    (0, 1): ['DTI'],
    (1, 0): ['TDI'],
    (1, 1): list(se_combo_name2idx.keys()),
}
# ============================================================================================= #
# SAVING DATA STRUCTURES
data_structures = {}
# Graph data structures
data_structures['adj_mats_orig'] = adj_mats_orig
data_structures['degrees'] = degrees
data_structures['edge_type2dim'] = edge_type2dim
data_structures['edge_type2decoder'] = edge_type2decoder
data_structures['edge_types'] = edge_types
data_structures['num_edge_types'] = num_edge_types
data_structures['edge2name'] = edge2name
# Feature data structures
data_structures['num_feat'] = num_feat
data_structures['nonzero_feat'] = nonzero_feat
data_structures['feat'] = feat
# Dictionaries
data_structures['gene2idx'] = gene2idx
data_structures['drug2idx'] = drug2idx
data_structures['se_mono_name2idx'] = se_mono_name2idx
data_structures['se_combo_name2idx'] = se_combo_name2idx
# Exporting
filename_out = './data_structures/DECAGON/DECAGON_' + words[2] + DSE*('_DSE_'+str(n_se_mono)) +\
BDM*'_BDM' + '_genes_' + str(n_genes) + '_drugs_' + str(n_drugs) + '_se_' + str(n_se_combo)
print('Output data exported in: ',filename_out)

with open(filename_out, 'wb') as f:
    pickle.dump(data_structures, f, protocol=2)

