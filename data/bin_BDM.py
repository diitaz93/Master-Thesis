#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# DECAGON_struct.py                                                                             #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/10/2020                                                                     #
# ============================================================================================= #
"""
Transforms the Algorithmic Complexity feature vectors (BDM) of the different adjacency matrices 
involved in DECAGON into sparse feature vectors. 
This is done setting two thresholds equal to the mean plus and minus one standard deviation of 
the complexity of each matrix and replacing the upper values with a 1, the lower values with a 
-1 and the ones in the middle with zeros.

Parameters
----------
n_genes : int
    number of genes in the network
n_drugs : int
    number of drugs in the network
n_se : int
    number of polypharmacy side effects

"""
# ============================================================================================= #
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle

parser = argparse.ArgumentParser(description='Network parameters for BDM file paths')
parser.add_argument('n_genes',type=str, help="Number of gene noodes in the network")
parser.add_argument('n_drugs',type=str, help="Number of drug noodes in the network")
parser.add_argument('n_se',type=str, help="Number of polypharmacy side effects")
args = parser.parse_args()
ppi_in_file = 'data_structures/BDM/PPI_BDM_real_genes_'+str(args.n_genes)
dti_in_file = 'data_structures/BDM/DTI_BDM_real_genes_'+str(args.n_genes)+'_drugs_'+\
              str(args.n_drugs)
ddi_in_file = 'data_structures/BDM/DDI_BDM_real_se_'+str(args.n_se)+'_drugs_'+str(args.n_drugs)
# ============================================================================================= #
# PPI
print('\n==== IMPORTED VARIABLES ====')
with open(ppi_in_file, 'rb') as f:
    PPI = pickle.load(f)
    for key in PPI.keys():
        globals()[key]=PPI[key]
        print(key,"Imported successfully")
print('\n')
# Mean and std
m_nodes = np.mean(nodebdm_ppi)
s_nodes = np.std(nodebdm_ppi)
m_rem = np.mean(rem_edgebdm_ppi)
s_rem = np.std(rem_edgebdm_ppi)
print('Mean of node BDM is ',m_nodes,' and std is ', s_nodes)
print('Mean of remove edges BDM is ',m_rem,' and std is ', s_rem,'\n')
# Up and down thresholds
d_thr_nodes = m_nodes-s_nodes
u_thr_nodes = m_nodes+s_nodes
d_thr_rem = m_rem-s_rem
u_thr_rem = m_rem+s_rem
# Node complexity sorting
neg_nodes = nodebdm_ppi<d_thr_nodes
pos_nodes = nodebdm_ppi>u_thr_nodes
bin_nodebdm_ppi = neg_nodes.astype(int)*-1+pos_nodes.astype(int)
# Edge complexity sorting
neg_rem = rem_edgebdm_ppi<d_thr_rem
pos_rem = rem_edgebdm_ppi>u_thr_rem
bin_rembdm_ppi = neg_rem.astype(int)*-1+pos_rem.astype(int)
# Filling proportion of vectors
sp_n = np.count_nonzero(bin_nodebdm_ppi)/len(nodebdm_ppi)
sp_r = np.count_nonzero(bin_rembdm_ppi)/len(rem_edgebdm_ppi)
print('The node feature vector is filled in a ',sp_n*100,'%')
print('The remove edge feature vector is filled in a ',sp_r*100,'%\n')
# SAVING
output_data = {}
output_data['nodebdm_ppi'] = bin_nodebdm_ppi
output_data['rem_edgebdm_ppi'] = bin_rembdm_ppi
output_data['vms_ppi'] = vms_ppi
output_data['rss_ppi'] = rss_ppi
output_data['time_ppi'] = time_ppi
output_data['jobs_ppi'] = jobs_ppi
# Compatibility with previous versions
if 'partition_type' in locals():
    output_data['partition_type'] = partition_type
words = ppi_in_file.split('_BDM_')
ppi_out_file = words[0] + '_BINBDM_' + words[1]
print('Output data exported in: ',ppi_out_file)
with open(ppi_out_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
# ============================================================================================= #
# DTI
print('\n==== IMPORTED VARIABLES ====')
with open(dti_in_file, 'rb') as f:
    DTI = pickle.load(f)
    for key in DTI.keys():
        globals()[key]=DTI[key]
        print(key,"Imported successfully")
print('\n')
# Mean and Standard deviation
m_nodes_drugs = np.mean(nodebdm_drugs_dti)
s_nodes_drugs = np.std(nodebdm_drugs_dti)
m_nodes_genes = np.mean(nodebdm_genes_dti)
s_nodes_genes = np.std(nodebdm_genes_dti)
m_rem_drugs = np.mean(rem_edgebdm_drugs_dti)
s_rem_drugs = np.std(rem_edgebdm_drugs_dti)
m_rem_genes = np.mean(rem_edgebdm_genes_dti)
s_rem_genes = np.std(rem_edgebdm_genes_dti)
print('Mean of drug node BDM is ',m_nodes_drugs,' and std is ', s_nodes_drugs)
print('Mean of gene node BDM is ',m_nodes_genes,' and std is ', s_nodes_genes)
print('Mean of remove drug edges BDM is ',m_rem_drugs,' and std is ', s_rem_drugs)
print('Mean of remove gene edges BDM is ',m_rem_genes,' and std is ', s_rem_genes,'\n')
# Up and down thresholds
d_thr_nodes_drugs = m_nodes_drugs-s_nodes_drugs
u_thr_nodes_drugs = m_nodes_drugs+s_nodes_drugs
d_thr_nodes_genes = m_nodes_genes-s_nodes_genes
u_thr_nodes_genes = m_nodes_genes+s_nodes_genes
d_thr_rem_drugs = m_rem_drugs-s_rem_drugs
u_thr_rem_drugs = m_rem_drugs+s_rem_drugs
d_thr_rem_genes = m_rem_genes-s_rem_genes
u_thr_rem_genes = m_rem_genes+s_rem_genes
# Node complexity sorting
neg_nodes_drugs = nodebdm_drugs_dti<d_thr_nodes_drugs
pos_nodes_drugs = nodebdm_drugs_dti>u_thr_nodes_drugs
bin_nodebdm_drugs_dti = neg_nodes_drugs.astype(int)*-1+pos_nodes_drugs.astype(int)
neg_nodes_genes = nodebdm_genes_dti<d_thr_nodes_genes
pos_nodes_genes = nodebdm_genes_dti>u_thr_nodes_genes
bin_nodebdm_genes_dti = neg_nodes_genes.astype(int)*-1+pos_nodes_genes.astype(int)
# Edge complexity sorting
neg_rem_drugs = rem_edgebdm_drugs_dti<d_thr_rem_drugs
pos_rem_drugs = rem_edgebdm_drugs_dti>u_thr_rem_drugs
bin_rembdm_drugs_dti = neg_rem_drugs.astype(int)*-1+pos_rem_drugs.astype(int)
neg_rem_genes = rem_edgebdm_genes_dti<d_thr_rem_genes
pos_rem_genes = rem_edgebdm_genes_dti>u_thr_rem_genes
bin_rembdm_genes_dti = neg_rem_genes.astype(int)*-1+pos_rem_genes.astype(int)
# Sparsity of vectors
sp_n_drugs = np.count_nonzero(bin_nodebdm_drugs_dti)/len(nodebdm_drugs_dti)
sp_r_drugs = np.count_nonzero(bin_rembdm_drugs_dti)/len(rem_edgebdm_drugs_dti)
print('The drug node feature vector is filled in a ',sp_n_drugs*100,'%')
print('The remove drug edge feature vector is filled in a ',sp_r_drugs*100,'%')
sp_n_genes = np.count_nonzero(bin_nodebdm_genes_dti)/len(nodebdm_genes_dti)
sp_r_genes = np.count_nonzero(bin_rembdm_genes_dti)/len(rem_edgebdm_genes_dti)
print('The gene node feature vector is filled in a ',sp_n_genes*100,'%')
print('The remove drug gene feature vector is filled in a ',sp_r_genes*100,'%\n')
# SAVING 
output_data = {}
output_data['nodebdm_drugs_dti'] = bin_nodebdm_drugs_dti
output_data['nodebdm_genes_dti'] = bin_nodebdm_genes_dti
output_data['rem_edgebdm_drugs_dti'] = bin_rembdm_drugs_dti
output_data['rem_edgebdm_genes_dti'] = bin_rembdm_genes_dti
output_data['vms_dti'] = vms_dti
output_data['rss_dti'] = vms_dti
output_data['time_dti'] = time_dti
output_data['jobs_dti'] = jobs_dti
# Compatibility with previous versions
if 'partition_type' in locals():
    output_data['partition_type'] = partition_type
words = dti_in_file.split('_BDM_')
dti_out_file = words[0] + '_BINBDM_' + words[1]
print('Output data exported in: ', dti_out_file)
with open(dti_out_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
# ============================================================================================= #
# DDI
print('\n==== IMPORTED VARIABLES ====')
with open(ddi_in_file, 'rb') as f:
    DDI = pickle.load(f)
    for key in DDI.keys():
        globals()[key]=DDI[key]
        print(key,"Imported successfully")
print('\n')
# Means and std
n_se = len(nodebdm_ddi_list)
m_nodes = np.mean(nodebdm_ddi_list,axis=1)
m_rem = np.mean(rem_edgebdm_ddi_list,axis=1)
s_nodes = np.std(nodebdm_ddi_list,axis=1)
s_rem = np.std(rem_edgebdm_ddi_list,axis=1)
# Up & down thresholds
d_thr_nodes = m_nodes-s_nodes
u_thr_nodes = m_nodes+s_nodes
d_thr_rem = m_rem-s_rem
u_thr_rem = m_rem+s_rem
bin_nodebdm_ddi_list = []
bin_rem_edgebdm_ddi_list = []
# Complexity sorting
for i in range(n_se):
    neg_nodes = nodebdm_ddi_list[i]<d_thr_nodes[i]
    pos_nodes = nodebdm_ddi_list[i]>u_thr_nodes[i]
    bin_nodebdm_ddi_list.append(neg_nodes.astype(int)*-1+pos_nodes.astype(int))
    neg_rem = rem_edgebdm_ddi_list[i]<d_thr_rem[i]
    pos_rem = rem_edgebdm_ddi_list[i]>u_thr_rem[i]
    bin_rem_edgebdm_ddi_list.append(neg_rem.astype(int)*-1+pos_rem.astype(int))
# Sparsity of vectors
L_inv = 1/len(nodebdm_ddi_list[0])
nm = np.mean(L_inv*np.count_nonzero(bin_nodebdm_ddi_list,axis=1))
rm = np.mean(L_inv*np.count_nonzero(bin_rem_edgebdm_ddi_list,axis=1))
norm = len(bin_nodebdm_ddi_list[0])*n_se
print('The node feature vectors are filled in average a ',nm*100,'%')
print('The remove edge feature vectors are filled in average a ',rm*100,'%')
# SAVING
output_data = {}
output_data['nodebdm_ddi_list'] = bin_nodebdm_ddi_list
output_data['rem_edgebdm_ddi_list'] = bin_rem_edgebdm_ddi_list
output_data['vms_ddi'] = vms_ddi
output_data['rss_ddi'] = rss_ddi
output_data['time_ddi'] = time_ddi
output_data['jobs_ddi'] = jobs_ddi
# Compatibility with previous versions
if 'partition_type' in locals():
    output_data['partition_type'] = partition_type
words = ddi_in_file.split('_BDM_')
ddi_out_file = words[0] + '_BINBDM_' + words[1]
print('Output data exported in: ', ddi_out_file)
with open(ddi_out_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)

