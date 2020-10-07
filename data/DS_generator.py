#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# DS_generator.py                                                                               #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 03/10/2020                                                                     #
# ============================================================================================= #
"""
Imports original DECAGON database and translates it into adjaceny matrices and enumeration 
dictionaries. First the original dataset is filtered so it has no unlinked nodes creating a 
consistent network. Then a fraction of the dataset is chosen, selecting a fixed number of 
polypharmacy side effects given by parameter N (defaults to 964). With the reduced network, 
the adjacency matrices and the node enumeration dictionaries are created and exported as a 
pickle python3 readable file.

Parameters
----------
number of side effects : int, default=964
    Number of joint drug side effects to be chosen from the complete dataset. If not given, 
    the program uses the maximum number of side effects used by the authors of DECAGON.
"""
# ============================================================================================= #
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from joblib import Parallel, delayed
parser = argparse.ArgumentParser(description='Remove outliers from datasets')
parser.add_argument('N', nargs='?',default =964,type=int, help="Number of side effects")
args = parser.parse_args()
N = args.N
# Import databases as pandas dataframes
PPI = pd.read_csv('original_data/bio-decagon-ppi.csv',sep=',')
DTI = pd.read_csv('original_data/bio-decagon-targets-all.csv',sep=',')
DDI = pd.read_csv('original_data/bio-decagon-combo.csv',sep=',')
DSE = pd.read_csv('original_data/bio-decagon-mono.csv',sep=',')
print('\nData loaded\n')
# Original number of interactions
orig_ppi = len(PPI.index)
orig_dti = len(DTI.index)
orig_ddi = len(DDI.index)
orig_dse = len(DSE.index)
# ============================================================================================= #
# REMOVING OUTLIERS
# PPI genes
PPI_genes = pd.unique(np.hstack((PPI['Gene 1'].values,PPI['Gene 2'].values)))
orig_genes_ppi = len(PPI_genes) # Original number of genes
# REDUCE DDI AND DSE DATABASES TO COMMON DRUGS ONLY
# DDI drugs
DDI_drugs = pd.unique(DDI[["STITCH 1", "STITCH 2"]].values.ravel())
orig_drugs_ddi = len(DDI_drugs) # Original number of drugs
orig_se_combo = len(pd.unique(DDI['Polypharmacy Side Effect'].values))
# Drugs with single side effects
DSE_drugs = pd.unique(DSE['STITCH'].values)
orig_drug_dse = len(DSE_drugs) # Original number of drugs
orig_se_mono = len(pd.unique(DSE['Side Effect Name']))
# Calculate the instersection of the DDI and DSE
# (i.e., the drugs in the interaction network that have single side effect)
inter_drugs = np.intersect1d(DDI_drugs,DSE_drugs,assume_unique=True)
# Choose only the entries in DDI that are in the intersection
DDI = DDI[np.logical_and(DDI['STITCH 1'].isin(inter_drugs).values,
                     DDI['STITCH 2'].isin(inter_drugs).values)]
# Some drugs in DDI that are common to all 3 datasets may only interact with genes that are
# non-common (outsiders). That is why we need to filter a second time using this array.
DDI_drugs = pd.unique(DDI[["STITCH 1", "STITCH 2"]].values.ravel())
DSE = DSE[DSE['STITCH'].isin(DDI_drugs)]
new_drugs_ddi = len(pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel()))
new_drugs_dse = len(pd.unique(DSE['STITCH'].values))
new_se_combo = len(pd.unique(DDI['Polypharmacy Side Effect'].values))
new_se_mono = len(pd.unique(DSE['Side Effect Name']))
# SELECT ONLY ENTRIES FROM DTI DATABASE THAT ARE PRESENT IN PREVIOUSLY REDUCED DATABASES
orig_genes_dti = len(pd.unique(DTI['Gene'].values))
orig_drugs_dti = len(pd.unique(DTI['STITCH'].values))
DTI = DTI[np.logical_and(DTI['STITCH'].isin(DDI_drugs),DTI['Gene'].isin(PPI_genes))]
DTI_genes = pd.unique(DTI['Gene'].values)
new_genes_dti = len(DTI_genes)
new_drugs_dti = len(pd.unique(DTI['STITCH'].values))
PPI = PPI[np.logical_or(PPI['Gene 1'].isin(DTI_genes),PPI['Gene 2'].isin(DTI_genes))]
print('Outliers removed\n')
# ============================================================================================= #
# REDUCED DATA STRUCTURES
# Choosing side effects. Sort DDI to be consistent with the authors
DDI['freq'] = DDI.groupby('Polypharmacy Side Effect')['Polypharmacy Side Effect']\
            .transform('count')
DDI = DDI.sort_values(by=['freq'], ascending=False).drop(columns=['freq'])
se = pd.unique(DDI['Polypharmacy Side Effect'].values)
se = se[:N]
# DDI
DDI = DDI[DDI['Polypharmacy Side Effect'].isin(se)].reset_index(drop=True)
DDI_drugs = pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel()) # Unique drugs 
drug2idx = {drug: i for i, drug in enumerate(DDI_drugs)}
se_names = pd.unique(DDI['Side Effect Name']) # Unique joint side effects
se_combo_name2idx = {se: i for i, se in enumerate(se_names)}
n_drugs = len(DDI_drugs)
# DSE
DSE = DSE[DSE['STITCH'].isin(DDI_drugs)].reset_index(drop=True)
dse_drugs = len(pd.unique(DSE['STITCH'].values))
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
PPI_genes = pd.unique(PPI[['Gene 1','Gene 2']].values.ravel()) # Unique genes in PPI
gene2idx = {gene: i for i, gene in enumerate(PPI_genes)}
n_genes = len(PPI_genes)
print('Side effects selected\n')
# ============================================================================================= #
# ADJACENCY MATRICES AND DEGREES
# DDI
def se_adj_matrix(se_name):
    m = np.zeros([n_drugs,n_drugs],dtype=int)
    seDDI = DDI[DDI['Side Effect Name'].str.match(se_name)].reset_index()
    for j in seDDI.index:
        row = drug2idx[seDDI.loc[j,'STITCH 1']]
        col = drug2idx[seDDI.loc[j,'STITCH 2']]
        m[row,col] = m[col,row] = 1
    return sp.csr_matrix(m) 
ddi_adj_list = Parallel(n_jobs=8)\
    (delayed(se_adj_matrix)(d) for d in se_combo_name2idx.keys())        
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
print('DDI adjacency matrices generated\n')
# DTI
dti_adj = np.zeros([n_genes,n_drugs],dtype=int)
for i in DTI.index:
    row = gene2idx[DTI.loc[i,'Gene']]
    col = drug2idx[DTI.loc[i,'STITCH']]
    dti_adj[row,col] = 1
dti_adj = sp.csr_matrix(dti_adj)
print('DTI adjacency matrix generated\n')
# PPI
ppi_adj = np.zeros([n_genes,n_genes],dtype=int)
for i in PPI.index:
    row = gene2idx[PPI.loc[i,'Gene 1']]
    col = gene2idx[PPI.loc[i,'Gene 2']]
    ppi_adj[row,col]=ppi_adj[col,row]=1
ppi_degrees = np.sum(ppi_adj,axis=0)
ppi_adj = sp.csr_matrix(ppi_adj)
print('PPI adjacency matrix generated\n')
# Drug feature matrix
drug_feat = np.zeros([n_drugs,n_semono],dtype=int)
for i in DSE.index:
    row = drug2idx[DSE.loc[i,'STITCH']]
    col = se_mono_name2idx[DSE.loc[i,'Side Effect Name']]
    drug_feat[row,col] = 1
drug_feat = sp.csr_matrix(drug_feat)
print('Drug feature matrix generated\n')
# ============================================================================================= #
# CONTROL PRINTING
# Interactions (edges)
print('==== CHANGES MADE IN DATA ====')
print('Interactions (edges)')
print ('Original number of PPI interactions',orig_ppi)
print ('New number of PPI interactions',len(PPI.index))
print('\n')
print ('Original number of DTI interactions',orig_dti)
print ('New number of DTI interactions',len(DTI.index))
print('\n')
print ('Original number of DDI interactions',orig_ddi)
print ('New number of DDI interactions', len(DDI.index))
print('\n')
print ('Original number of DSE interactions',orig_dse)
print('New number of DSE interactions',len(DSE.index))
print('\n')
# Drugs and genes (nodes)
print('Drugs and genes (nodes)')
print("Original number of drugs in DSE:",orig_drug_dse)
print("New number of drugs in DSE:", dse_drugs)
print('\n')
print("Original number drugs in DTI",orig_drugs_dti)
print("New number of drugs in DTI",dti_drugs)
print('\n')
print('Original number of genes in DTI:',orig_genes_dti)
print('New number of genes in DTI:',dti_genes)
print('\n')
print('Original number of genes in PPI:',orig_genes_ppi)
print('New number of genes in PPI:',n_genes)
print('\n')
print('Original number of drugs in DDI:',orig_drugs_ddi)
print('New number of drugs in DDI:',n_drugs)
print('\n')
# Side effects
print('Side effects')
print('Original number of joint side effects:',orig_se_combo)
print('New number of joint side effects:', len(se_names))
print('\n')
print('Original number of single side effects:', orig_se_mono)
print('New number of single side effects:', n_semono)
# ============================================================================================= #
# SAVING DATA STRUCTURES
data = {}
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
# Exporting
filename = './data_structures/DS/DS_real_DSE_' + str(n_semono) +\
           '_genes_' + str(n_genes) + '_drugs_' + str(n_drugs) + '_se_' + str(N)
print('Output_file: ',filename,'\n')
with open(filename, 'wb') as f:
    pickle.dump(data, f, protocol=3)


