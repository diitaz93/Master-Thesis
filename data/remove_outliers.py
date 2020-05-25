#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# remove_outliers.py                                                                            #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 07/05/2020                                                                     #
# ============================================================================================= #
"""
Imports original DECAGON database + protein features and writes out new data files containing a 
consistent database. The new database has all the unlinked nodes (outliers) removed. In addition,
normalizes the protein features corresponding to the number of helices, strands and turns to the
average and median values of the protein length. All of the nodes (genes, drugs) from the DTI 
database are included in their respective interaction databases (PPI, PF; and DDI, DSE 
respectively) but not necessarily the opposite. 
"""
# ============================================================================================= #
import numpy as np
import pandas as pd
# Import databases as pandas dataframes
PPI = pd.read_csv('original_data/bio-decagon-ppi.csv',sep=',')
PF = pd.read_csv('original_data/proteins.csv',sep=';')
DTI = pd.read_csv('original_data/bio-decagon-targets-all.csv',sep=',')
DDI = pd.read_csv('original_data/bio-decagon-combo.csv',sep=',')
DSE = pd.read_csv('original_data/bio-decagon-mono.csv',sep=',')
# Remove comma after thousand
PF['Mass'] = PF['Mass'].apply(lambda x: x.replace(',', '')).astype('int')
# Original number of interactions
orig_ppi = len(PPI.index)
orig_dti = len(DTI.index)
orig_pf = len(PF.index)
orig_ddi = len(DDI.index)
orig_dse = len(DSE.index)
# ============================================================================================= #
# REDUCE PPI AND PF DATABASES TO COMMON GENES ONLY
# PPI genes
PPI_genes = pd.unique(np.hstack((PPI['Gene 1'].values,PPI['Gene 2'].values))) #int
orig_genes_ppi = len(PPI_genes) # Original number of genes
# PF genes
PF_genes = pd.unique(PF['GeneID'].apply(int)) #int
orig_genes_pf = len(PF_genes) # Original number of genes
# Calculate the instersection of the PPI and PF
# (i.e., the genes in the interaction network that code proteins with features)
inter_genes = np.intersect1d(PPI_genes,PF_genes,assume_unique=True) #int
# Chooses only the entries in PPI that are in the intersection
PPI = PPI[np.logical_and(PPI['Gene 1'].isin(inter_genes).values,
                     PPI['Gene 2'].isin(inter_genes).values)]
# Some genes in PPI that are common to all 3 datasets may only interact with genes that are
# non-common (outsiders). That is why we need to filter a second time using this array.
PPI_genes = pd.unique(np.hstack((PPI['Gene 1'].values,PPI['Gene 2'].values)))
PF = PF[PF['GeneID'].apply(int).isin(PPI_genes)]
new_genes_ppi = len(pd.unique(PPI[["Gene 1", "Gene 2"]].values.ravel()))
new_genes_pf = len(pd.unique(PF['GeneID'].values))
new_ppi = len(PPI.index)
new_pf = len(PF.index)
# ============================================================================================= #
# REDUCE DDI AND DSE DATABASES TO COMMON DRUGS ONLY
# DDI drugs
DDI_drugs = pd.unique(DDI[["STITCH 1", "STITCH 2"]].values.ravel())
orig_drugs_ddi = len(DDI_drugs) # Original number of drugs
# Drugs with single side effects
DSE_drugs = pd.unique(DSE['STITCH'].values)
orig_drugs_dse = len(DSE_drugs) # Original number of drugs
# Calculate the instersection of the DDI and DSE
# (i.e., the drugs in the intercation network that have single side effect)
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
new_ddi = len(DDI.index)
new_dse = len(DSE.index)
# ============================================================================================= #
# SELECT ONLY ENTRIES FROM DTI DATABASE THAT ARE PRESENT IN PREVIOUS REDUCED DATABASES
orig_genes_dti = len(pd.unique(DTI['Gene'].values))
orig_drugs_dti = len(pd.unique(DTI['STITCH'].values))
DTI = DTI[np.logical_and(DTI['STITCH'].isin(DDI_drugs),DTI['Gene'].isin(PPI_genes))]
new_dti = len(DTI.index)
new_genes_dti = len(pd.unique(DTI['Gene'].values))
new_drugs_dti = len(pd.unique(DTI['STITCH'].values))
# ============================================================================================= #
# NORMALIZES PROTEIN FEATURES
avg = PF['Length'].mean()
med = PF['Length'].median()
norm_strand = PF['n_strands']/med
norm_strand_med = norm_strand/norm_strand.max()
norm_strand = PF['n_strands']/avg
norm_strand_avg = norm_strand/norm_strand.max()
norm_helix = PF['n_helices']/med
norm_helix_med = norm_helix/norm_helix.max()
norm_helix = PF['n_helices']/avg
norm_helix_avg = norm_helix/norm_helix.max()
norm_turns = PF['n_turns']/med
norm_turns_med = norm_turns/norm_turns.max()
norm_turns = PF['n_turns']/avg
norm_turns_avg = norm_turns/norm_turns.max()
PF['Normalized Helices(Mean)'] = norm_helix_avg
PF['Normalized Helices(Median)'] = norm_helix_med
PF['Normalized Strands(Mean)'] = norm_strand_avg
PF['Normalized Strands(Median)'] = norm_strand_med
PF['Normalized Turns(Mean)'] = norm_turns_avg
PF['Normalized Turns(Median)'] = norm_turns_med
# ============================================================================================= #
# CONTROL PRINTING
print ('Original number of PPI interactions',orig_ppi)
print ('New number of PPI interactions',new_ppi)
print('\n')
print ('Original number of DTI interactions',orig_dti)
print ('New number of DTI interactions',new_dti)
print('\n')
print ('Original number of DDI interactions',orig_ddi)
print ('New number of DDI interactions',new_ddi)
print('\n')
print ('Original number of proteins with features',orig_pf)
print ('New number of proteins with features',new_pf)
print('\n')
print ('Original number of single side effect interactions',orig_dse)
print('New number of single side effect interactions',new_dse)
print('\n')
print("Original number of unique genes in PPI:",orig_genes_ppi)
print("New number of unique genes in PPI:",new_genes_ppi)
print("Original number of genes whose proteins have features:",orig_genes_pf)
print("New number of genes whose proteins have features:",new_genes_pf)
print("Original number of unique genes in DTI",orig_genes_dti)
print("New number of unique genes in DTI",new_genes_dti)
print('\n')
print("Original number of unique drugs in DDI:",orig_drugs_ddi)
print("New number of unique drugs in DDI:",new_drugs_ddi)
print("Original number of drugs with single side effects:",orig_drugs_dse)
print("New number of drugs with single side effects:",new_drugs_dse)
print("Original number of unique drugs in DTI",orig_drugs_dti)
print("New number of unique drugs in DTI",new_drugs_dti)
# ============================================================================================= #
# EXPORTING DATABASE TO CSV FILES
PPI.to_csv('./clean_data/new-decagon-ppi.csv',index=False,sep=',')
DTI.to_csv('./clean_data/new-decagon-targets.csv',index=False,sep=',')
DDI.to_csv('./clean_data/new-decagon-combo.csv',index=False,sep=',')
DSE.to_csv('./clean_data/new-decagon-mono.csv',index=False,sep=',')
PF.to_csv('./clean_data/new-decagon-genes.csv',index=False,sep=',')

