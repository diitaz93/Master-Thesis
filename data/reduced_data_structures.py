#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# reduced_data_structures.py                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Imports the treated (outlier-free) dataset and samples a consistent small subset to be run in 
small machines. The generated dataset is limited to a number of drug-drug interactions (side 
effects) specified by the variable N entered as argument. From the reduced dataset, it creates 
the needed data structures to be used in BDM calculations and DECAGON training.

Parameters
----------
number of side effects : int
    Number of joint drug side effects to be chosen from the complete dataset.

"""
# ============================================================================================= #
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shelve
N = int(sys.argv[1]) # Number of side effects
# Import databases as pandas dataframes
PPI = pd.read_csv('clean_data/new-decagon-ppi.csv',sep=',')
PF = pd.read_csv('clean_data/new-decagon-genes.csv',sep=',')
DTI = pd.read_csv('clean_data/new-decagon-targets.csv',sep=',')
DDI = pd.read_csv('clean_data/new-decagon-combo.csv',sep=',')
DSE = pd.read_csv('clean_data/new-decagon-mono.csv',sep=',')
SE = pd.read_csv('original_data/bio-decagon-effectcategories.csv',sep=',')
# Number of interactions
orig_ppi = len(PPI.index)
orig_pf = len(PF.index)
orig_dti = len(DTI.index)
orig_ddi = len(DDI.index)
orig_dse = len(DSE.index)
# Number of nodes
orig_ddi_drugs = len(pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel()))
orig_ppi_genes = len(pd.unique(PPI[['Gene 1','Gene 2']].values.ravel()))
orig_dti_drugs = len(pd.unique(DTI['STITCH']))
orig_dti_genes = len(pd.unique(DTI['Gene']))
orig_se_mono = len(pd.unique(DSE['Side Effect Name']))
# ============================================================================================= #
# SAMPLING DATASET AND INDEX DATA STRUCTURES
se = SE.sample(n=N, axis=0)['Side Effect'].values # Choosing side effects
# DDI
DDI = DDI[DDI['Polypharmacy Side Effect'].isin(se)].reset_index(drop=True)
DDI_drugs = pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel()) # Unique drugs 
drug2idx = {drug: i for i, drug in enumerate(DDI_drugs)}
se_names = pd.unique(DDI['Side Effect Name']) # Unique joint side effects
se_combo_name2idx = {se: i for i, se in enumerate(se_names)}
n_drugs = len(DDI_drugs)
# DSE
DSE = DSE[DSE['STITCH'].isin(DDI_drugs)].reset_index(drop=True)
se_mono_names = pd.unique(DSE['Side Effect Name'].values) # Unique individual side effects
se_mono_name2idx = {name: i for i, name in enumerate(se_mono_names)}
n_semono = len(se_mono_names)
# DTI
DTI = DTI[DTI['STITCH'].isin(DDI_drugs)].reset_index(drop=True)
DTI_genes = pd.unique(DTI['Gene']) # Unique genes in DTI
DTI_drugs = pd.unique(DTI['STITCH']) # Unique drugs in DTI
dti_drugs = len(DTI_drugs)
dti_genes = len(DTI_genes)
# PPI
PPI = PPI[np.logical_or(PPI['Gene 1'].isin(DTI_genes),
                       PPI['Gene 2'].isin(DTI_genes))].reset_index(drop=True)
PPI_genes = pd.unique(PPI[['Gene 1','Gene 2']].values.ravel()) # Unique genes is PPI
gene2idx = {gene: i for i, gene in enumerate(PPI_genes)}
n_genes = len(PPI_genes)
# Protein features
PF = PF[PF['GeneID'].isin(PPI_genes)].reset_index(drop=True)
# ============================================================================================= #
# ADJACENCY MATRICES AND DEGREES
# DDI
ddi_adj_list = []
for i in se_combo_name2idx.keys():
    m = np.zeros([n_drugs,n_drugs],dtype=int)
    seDDI = DDI[DDI['Side Effect Name'].str.match(i)].reset_index()
    for j in seDDI.index:
        row = drug2idx[seDDI.loc[j,'STITCH 1']]
        col = drug2idx[seDDI.loc[j,'STITCH 2']]
        m[row,col] = m[col,row] = 1
    ddi_adj_list.append(sp.csr_matrix(m))
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
# DTI
dti_adj = np.zeros([n_genes,n_drugs],dtype=int)
for i in DTI.index:
    row = gene2idx[DTI.loc[i,'Gene']]
    col = drug2idx[DTI.loc[i,'STITCH']]
    dti_adj[row,col] = 1
dti_adj = sp.csr_matrix(dti_adj)
# PPI
ppi_adj = np.zeros([n_genes,n_genes],dtype=int)
for i in PPI.index:
    row = gene2idx[PPI.loc[i,'Gene 1']]
    col = gene2idx[PPI.loc[i,'Gene 2']]
    ppi_adj[row,col]=ppi_adj[col,row]=1
ppi_degrees = np.sum(ppi_adj,axis=0)
ppi_adj = sp.csr_matrix(ppi_adj)
# Drug feature matrix
drug_feat = np.zeros([n_drugs,n_semono],dtype=int)
for i in DSE.index:
    row = drug2idx[DSE.loc[i,'STITCH']]
    col = se_mono_name2idx[DSE.loc[i,'Side Effect Name']]
    drug_feat[row,col] = 1
drug_feat = sp.csr_matrix(drug_feat)
# Protein feature matrices
prot_feat = sp.coo_matrix(
    PF[['GeneID', 'Length', 'Mass', 'n_helices', 'n_strands', 'n_turns']].to_numpy())
norm_prot_feat = sp.coo_matrix(
    PF[['Normalized Helices(Mean)', 'Normalized Helices(Median)',
       'Normalized Strands(Mean)', 'Normalized Strands(Median)',
       'Normalized Turns(Mean)', 'Normalized Turns(Median)']].to_numpy())
# ============================================================================================= #
# CONTROL PRINTING
print('Original number of PPI interactions:', orig_ppi)
print('New number of PPI interactions:', len(PPI.index))
print('\n')
print('Original number of DDI interactions:', orig_ddi)
print('New number of DDI interactions:', len(DDI.index))
print('\n')
print('Original number of DTI interactions:', orig_dti)
print('New number of DTI interactions:', len(DTI.index))
print('Original number of DTI genes:', orig_dti_genes)
print('New number of DTI genes:',len(pd.unique(DTI['Gene'].values)))
print('Original number of DTI drugs:', orig_dti_drugs)
print('New number of DTI drugs:',len(pd.unique(DTI['STITCH'].values)))
print('\n')
print('Original number of DSE interactions:', orig_dse)
print('New number of DSE interactions:', len(DSE.index))
print('\n')
print('Original number of genes:',orig_ppi_genes)
print('New number of genes:', n_genes)
print('\n')
print('Original number of drugs:',orig_ddi_drugs)
print('New number of drugs:', n_drugs)
print('Original number of single side effects:', orig_se_mono)
print('New number of single side effects:', n_semono)
print('\n')
print('Original number of proteins with features:', orig_pf)
print('New number of proteins with features:', len(PF.index))
print('\n')
# ============================================================================================= #
# SAVING DATA STRUCTURES
data = shelve.open('./data_structures/decagon','n',protocol=2)
# Dictionaries
data['gene2idx'] = gene2idx
data['drug2idx'] = drug2idx
data['se_mono_name2idx'] = se_mono_name2idx
data['se_combo_name2idx'] = se_combo_name2idx
# DDI
data['ddi_adj_list'] = ddi_adj_list
data['ddi_degrees_list'] = ddi_degrees_list
# DTI
data['dti_adj'] = dti_adj
# PPI
data['ppi_adj'] = ppi_adj
data['ppi_degrees'] = ppi_degrees
# DSE
data['drug_feat'] = drug_feat
# PF
data['prot_feat'] = prot_feat
data['norm_prot_feat'] = norm_prot_feat
data.close()
