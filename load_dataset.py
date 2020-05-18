
#Python 3.7/2.7
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from itertools import combinations, chain
import shelve
from getpass import getuser
from data.load_functions import *


# Loading Gene data (PPI)
ppi, gene2idx = load_ppi(fname='data/clean_data/ppi_mini.csv')
ppi_adj = nx.adjacency_matrix(ppi)
ppi_degrees = np.array(ppi_adj.sum(axis=0)).squeeze() 
ppi_genes = ppi.number_of_nodes() # Number of genes (nodes)
# Loading individual side effects
stitch2se, semono2name, semono2idx = load_mono_se(fname='data/clean_data/mono_mini.csv')
n_semono = len(semono2name)
print('Number of individual side effects: ', n_semono)
# Loading Target data (DTI)
stitch2proteins = load_targets(fname='data/clean_data/targets_mini.csv')
dti_drugs = len(pd.unique(stitch2proteins.keys()))
dti_genes = len(set(chain.from_iterable(stitch2proteins.itervalues())))
print('Number of genes in DTI:', dti_genes)
print('Number of drugs in DTI:', dti_drugs)
# Loading Drug data (DDI)
combo2stitch, combo2se, secombo2name, drug2idx = load_combo_se(fname='data/clean_data/combo_mini.csv')
n_secombo = len(secombo2name)
# Loading Side effect data (features)
stitch2se, semono2name, semono2idx = load_mono_se(fname='data/clean_data/mono_mini.csv')
# Loading protein features
PF = pd.read_csv('data/clean_data/genes_mini.csv', sep=',',header=None).to_numpy()
ddi_drugs = len(drug2idx)
print('Number of drugs: ', ddi_drugs)

username = getuser()
filename = 'session_' + username + '_NSE_' + str(n_secombo)
print(filename)
my_shelf = shelve.open(filename,'n')
my_shelf['ppi_adj'] = ppi_adj

# Drug-target adjacency matrix
dti_adj = np.zeros([ppi_genes,ddi_drugs],dtype=int)
for drug in drug2idx.keys():
    for gene in stitch2proteins[drug]:
        if gene==set():
            continue
        else:
            idp = gene2idx[str(gene)]
            idd = drug2idx[drug]
            dti_adj[idp,idd] = 1
dti_adj = sp.csr_matrix(dti_adj)
my_shelf['dti_adj'] = dti_adj

# DDI adjacency matrix
ddi_adj_list = []
for se in secombo2name.keys():
    m = np.zeros([ddi_drugs,ddi_drugs],dtype=int)
    for pair in combo2se.keys():
        if se in combo2se[pair]:
            d1,d2 = combo2stitch[pair]
            m[drug2idx[d1],drug2idx[d2]] = m[drug2idx[d2],drug2idx[d1]] = 1
    ddi_adj_list.append(sp.csr_matrix(m))    
ddi_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in ddi_adj_list]
my_shelf['ddi_adj_list'] = ddi_adj_list
np.save('adjs.npy',ddi_adj_list)
my_shelf.close()
