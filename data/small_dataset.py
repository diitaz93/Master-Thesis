#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# ============================================================================================= #
# small_dataset.py                                                                            #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Imports the treated (outlier-free) dataset and samples a consistent small subset to be run in small machines. The generated dataset is limited to a number of drug-drug interactions (side effects) specified by the variable N entered as argument. 
"""
# ============================================================================================= #
from __future__ import print_function # Only for Python 2
import sys
import numpy as np
import pandas as pd
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
orig_drugs = len(pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel()))
orig_genes = len(pd.unique(PPI[['Gene 1','Gene 2']].values.ravel()))
# ============================================================================================= #
# SAMPLING DATASET
se = SE.sample(n=N, axis=0)['Side Effect'].values # Choosing side effects
# DDI
DDI = DDI[DDI['Polypharmacy Side Effect'].isin(se)].reset_index(drop=True)
DDI_drugs = pd.unique(DDI[['STITCH 1','STITCH 2']].values.ravel())
new_drugs = len(DDI_drugs)
# DSE
DSE = DSE[DSE['STITCH'].isin(DDI_drugs)].reset_index(drop=True)
# DTI
DTI = DTI[DTI['STITCH'].isin(DDI_drugs)].reset_index(drop=True)
DTI_genes = pd.unique(DTI['Gene'])
# PPI
PPI = PPI[np.logical_or(PPI['Gene 1'].isin(DTI_genes),
                       PPI['Gene 2'].isin(DTI_genes))].reset_index(drop=True)
PPI_genes = pd.unique(PPI[['Gene 1','Gene 2']].values.ravel())
new_genes = len(PPI_genes)
# Protein features
PF = PF[PF['GeneID'].isin(PPI_genes)].reset_index(drop=True)
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
print('New number of DTI genes:',len(pd.unique(DTI['Gene'].values)))
print('New number of DTI drugs:',len(pd.unique(DTI['STITCH'].values)))
print('\n')
print('Original number of DSE interactions:', orig_dse)
print('New number of DSE interactions:', len(DSE.index))
print('\n')
print('Original number of proteins with features:', orig_pf)
print('New number of proteins with features:', len(PF.index))
print('\n')
print('Original number of genes:',orig_genes)
print('New number of genes:', new_genes)
print('\n')
print('Original number of drugs:',orig_drugs)
print('New number of drugs:', new_drugs)
# ============================================================================================= #
# EXPORTING DATABASE TO CSV FILES
PPI.to_csv('./clean_data/ppi_mini.csv',header=False,index=False,sep=',')
DTI.to_csv('./clean_data/targets_mini.csv',header=False,index=False,sep=',')
DDI.to_csv('./clean_data/combo_mini.csv',header=False,index=False,sep=',')
DSE.to_csv('./clean_data/mono_mini.csv',header=False,index=False,sep=',')
PF.to_csv('./clean_data/genes_mini.csv',header=False,index=False,sep=',')