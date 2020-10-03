#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# remove_outliers.py                                                                            #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Imports original DECAGON database and writes out new data files containing a 
consistent database. The new database has all the unlinked nodes (outliers) removed. All of the
nodes (genes, drugs) from the DTI database are included in their respective interaction databases
(PPI, PF; and DDI, DSE respectively) but not necessarily the opposite.
"""
# ============================================================================================= #
import numpy as np
import pandas as pd
# Import databases as pandas dataframes
PPI = pd.read_csv('original_data/bio-decagon-ppi.csv',sep=',')
DTI = pd.read_csv('original_data/bio-decagon-targets-all.csv',sep=',')
DDI = pd.read_csv('original_data/bio-decagon-combo.csv',sep=',')
DSE = pd.read_csv('original_data/bio-decagon-mono.csv',sep=',')
# Original number of interactions
orig_ppi = len(PPI.index)
orig_dti = len(DTI.index)
orig_ddi = len(DDI.index)
orig_dse = len(DSE.index)
# ============================================================================================= #
# REDUCE PPI AND PF DATABASES TO COMMON GENES ONLY
# PPI genes
PPI_genes = pd.unique(np.hstack((PPI['Gene 1'].values,PPI['Gene 2'].values))) #int
orig_genes_ppi = len(PPI_genes) # Original number of genes
# ============================================================================================= #
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
# ============================================================================================= #
# SELECT ONLY ENTRIES FROM DTI DATABASE THAT ARE PRESENT IN PREVIOUS REDUCED DATABASES
orig_genes_dti = len(pd.unique(DTI['Gene'].values))
orig_drugs_dti = len(pd.unique(DTI['STITCH'].values))
DTI = DTI[np.logical_and(DTI['STITCH'].isin(DDI_drugs),DTI['Gene'].isin(PPI_genes))]
DTI_genes = pd.unique(DTI['Gene'].values)
new_genes_dti = len(DTI_genes)
new_drugs_dti = len(pd.unique(DTI['STITCH'].values))
PPI = PPI[np.logical_or(PPI['Gene 1'].isin(DTI_genes),PPI['Gene 2'].isin(DTI_genes))]
# ============================================================================================= #
# CONTROL PRINTING
# Interactions (edges)
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
print("New number of drugs in DSE:",new_drugs_dse)
print('\n')
print("Original number drugs in DTI",orig_drugs_dti)
print("New number of drugs in DTI",new_drugs_dti)
print('\n')
print('Original number of genes in DTI:',orig_genes_dti)
print('New number of genes in DTI:',new_genes_dti)
print('\n')
print('Original number of genes in PPI:',orig_genes_ppi)
print('New number of genes in PPI:',new_genes_ppi)
print('\n')
print('Original number of drugs in DDI:',orig_drugs_ddi)
print('New number of drugs in DDI:',new_drugs_ddi)
print('\n')
# Side effects
print('Side effects')
print('Original number of joint side effects:',orig_se_combo)
print('New number of joint side effects:', new_se_combo)
print('\n')
print('Original number of single side effects:', orig_se_mono)
print('New number of single side effects:', new_se_mono)
# ============================================================================================= #
# EXPORTING DATABASE TO CSV FILES
PPI.to_csv('./clean_data/new-decagon-ppi.csv',index=False,sep=',')
DTI.to_csv('./clean_data/new-decagon-targets.csv',index=False,sep=',')
DDI.to_csv('./clean_data/new-decagon-combo.csv',index=False,sep=',')
DSE.to_csv('./clean_data/new-decagon-mono.csv',index=False,sep=',')

