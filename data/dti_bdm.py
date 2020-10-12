#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# dti_bdm.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Calculates the algoritmic complexity of the gene-drug network of the DECAGON dataset. The 
dataset is given as a bipartite adjacency matrix of size 𝑁𝑔𝑒𝑛𝑒𝑠×𝑁𝑑𝑟𝑢𝑔𝑠. The code uses the 
package pybdm to calculate the complexity contribution of each node and its corresponding edges 
in both axis of the matrix separately. The calculated feature vectors are exported along with 
relevant data as a pickle readable format file.

Parameters
----------
path : string
    (Relative) path to the file of data structures.
"""
# ============================================================================================= #
import numpy as np
import time
import os
import sys
import psutil
import pickle
import warnings
from pybdm import BDM
from pybdm.partitions import PartitionRecursive
from algorithms import PerturbationExperiment, NodePerturbationExperiment
from getpass import getuser
# Settings and loading of adj matrix
input_file = str(sys.argv[1])
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
warnings.filterwarnings("ignore")
with open(input_file, 'rb') as f:
    dti_adj = pickle.load(f)['dti_adj']
print('\nInput data loaded\n')
jobs = 4
bdm = BDM(ndim=2, partition=PartitionRecursive)
part = 'PartitionRecursive'
# ============================================================================================= #
# CALCULATION
# Node perturbation
dti_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=True, 
                                         parallel=True,jobs=jobs)
dti_nodeper.set_data(np.array(dti_adj.todense()))
print("Initial BDM calculated for nodes\n")
nodebdm_genes_dti,nodebdm_drugs_dti = dti_nodeper.run()
print('BDM for DTI calculated\n')
# Edge perturbation
dti_edgeper = PerturbationExperiment(bdm, bipartite_network=True)
dti_edgeper.set_data(np.array(dti_adj.todense()))
print("Initial BDM calculated for edges\n")
rem_edgebdm_genes_dti, rem_edgebdm_drugs_dti = dti_edgeper.run_removing_edges()
print('Edge BDM for DTI calculated\n')
# ============================================================================================= #
# EXPORTING
genes,drugs = dti_adj.shape
memUse = ps.memory_info()
total_time=time.time()-start
output_data = {}
output_data['nodebdm_drugs_dti'] = nodebdm_drugs_dti
output_data['nodebdm_genes_dti'] = nodebdm_genes_dti
output_data['rem_edgebdm_drugs_dti'] = rem_edgebdm_drugs_dti
output_data['rem_edgebdm_genes_dti'] = rem_edgebdm_genes_dti
output_data['vms_dti'] = memUse.vms
output_data['rss_dti'] = memUse.rss
output_data['time_dti'] = total_time
output_data['jobs_dti'] = jobs
output_data['partition_type'] = part
path = os.getcwd()
words = input_file.split('_')
output_file = path + '/data_structures/BDM/DTI_BDM_' + words[2] + '_genes_' + str(genes) +\
             '_drugs_' + str(drugs)
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
print('Output data exported in ', output_file,'\n')
