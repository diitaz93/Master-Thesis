#Python 3

import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from itertools import combinations, chain
import shelve
from getpass import getuser
from pybdm import BDM
from node import NodePerturbationExperiment

#PF
PF = pd.read_csv('data/clean_data/genes_mini.csv',
                 sep=',',names=['GeneID','Length','Mass','n_helices','n_strands','n_turns'])
genes = pd.unique(PF['GeneID'].values)
gene2idx = {gene: i for i, gene in enumerate(genes)}
n_genes = len(gene2idx)
print('Number of genes in network:',n_genes)
prot_feat = sp.coo_matrix(PF.to_numpy())
print('Protein feature matrix calculated')
#PPI
PPI = pd.read_csv('data/clean_data/ppi_mini.csv',sep=',',names=["Gene_1", "Gene_2"])
# PPI adjacency matrix
ppi_adj = np.zeros([n_genes,n_genes],dtype=int)
for i in PPI.index:
    row = gene2idx[PPI.loc[i,'Gene_1']]
    col = gene2idx[PPI.loc[i,'Gene_2']]
    ppi_adj[row,col]=ppi_adj[col,row]=1
ppi_degrees = np.sum(ppi_adj,axis=0)
ppi_adj = sp.csr_matrix(ppi_adj)
print('PPI adjacency matrix and degrees calculated')
#DDI
DDI = pd.read_csv('data/clean_data/combo_mini.csv', sep=','
                  ,names=["STITCH_1", "STITCH_2", "SE", "SE_name"])
drugs = pd.unique(np.hstack((DDI['STITCH_1'].values,DDI['STITCH_2'].values)))
drug2idx = {drug: i for i, drug in enumerate(drugs)}
n_drugs = len(drug2idx)
print('Number of drugs in the network',n_drugs)
se_names = pd.unique(DDI['SE_name'].values)
se_combo_name2idx = {se: i for i, se in enumerate(se_names)}
n_secombo = len(se_combo_name2idx)
print('Number of DDI side effects',n_secombo)
# DDI adjacency matrices
ddi_adj_list = []
for i in se_combo_name2idx.keys():
    m = np.zeros([n_drugs,n_drugs],dtype=int)
    seDDI = DDI[DDI['SE_name'].str.match(i)].reset_index()
    for j in seDDI.index:
        row = drug2idx[seDDI.loc[j,'STITCH_1']]
        col = drug2idx[seDDI.loc[j,'STITCH_2']]
        m[row,col] = m[col,row] = 1
    ddi_adj_list.append(sp.csr_matrix(m))
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
print('DDI adjacency matrix list and degree list calculated')
#DTI
DTI = pd.read_csv('data/clean_data/targets_mini.csv',sep=',',names=["STITCH", "GENE"])
dti_drugs = len(pd.unique(DTI['STITCH'].values))
dti_genes = len(pd.unique(DTI['GENE'].values))
print('Number of DTI drugs:',dti_drugs)
print('Number of DTI genes:',dti_genes)
#DTI adjacency matrix
dti_adj = np.zeros([n_genes,n_drugs],dtype=int)
for i in DTI.index:
    row = gene2idx[DTI.loc[i,'GENE']]
    col = drug2idx[DTI.loc[i,'STITCH']]
    dti_adj[row,col] = 1
dti_adj = sp.csr_matrix(dti_adj)
print('DTI adjacency matrix calculated')
#DSE
DSE = pd.read_csv('data/clean_data/mono_mini.csv', sep=',',names=["STITCH","SE", "SE_name"])
se_mono_names = pd.unique(DSE['SE_name'].values)
se_mono_name2idx = {name: i for i, name in enumerate(se_mono_names)}
n_semono = len(se_mono_name2idx)
print('Number of DSE side effects:',n_semono)
# Drug Feature matrix
drug_feat = np.zeros([n_drugs,n_semono],dtype=int)
for i in DSE.index:
    row = drug2idx[DSE.loc[i,'STITCH']]
    col = se_mono_name2idx[DSE.loc[i,'SE_name']]
    drug_feat[row,col] = 1
drug_feat = sp.csr_matrix(drug_feat)
print('Drug feature matrix calculated')
#Save
data = shelve.open('./results/decagon','n',protocol=2)
#PF
data['prot_feat'] = prot_feat
data['gene2idx'] = gene2idx
#PPI
data['ppi_adj'] = ppi_adj
data['ppi_degrees'] = ppi_degrees
#DDI
data['se_mono_name2idx'] = se_mono_name2idx
data['ddi_adj_list'] = ddi_adj_list
data['ddi_degrees_list'] = ddi_degrees_list
data['drug2idx'] = drug2idx
#DTI
data['dti_drugs'] = dti_drugs
data['dti_genes'] = dti_genes
data['dti_adj'] = dti_adj
#DSE
data['drug_feat'] = drug_feat
data['se_mono_name2idx'] = se_mono_name2idx
data.close()
