#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# ppi_bdm.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Calculates the algoritmic complexity of the gene network of the DECAGON dataset. The dataset is 
given as an adjacency matrix of size 𝑁𝑔𝑒𝑛𝑒𝑠×𝑁𝑔𝑒𝑛𝑒𝑠. The code uses the package pybdm to 
calculate the complexity contribution of each node and its corresponding edges. The calculated 
feature vector along with relevant data are exported as a pickle readable format file.

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
    ppi_adj = pickle.load(f)['ppi_adj']
print('\nInput data loaded\n')
jobs = 32
bdm = BDM(ndim=2, partition=PartitionRecursive)
part = 'PartitionRecursive'
# ============================================================================================= #
# CALCULATION
# Node perturbation
ppi_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                         parallel=True,jobs=jobs)
ppi_nodeper.set_data(np.array(ppi_adj.todense()))
print("Initial BDM calculated for nodes\n")
nodebdm_ppi = ppi_nodeper.run()
print('Node BDM for PPI calculated\n')
# Edge perturbation
ppi_edgeper = PerturbationExperiment(bdm, bipartite_network=False)
ppi_edgeper.set_data(np.array(ppi_adj.todense()))
print("Initial BDM calculated for nodes\n")
rem_edgebdm_ppi = ppi_edgeper.run_removing_edges()
print('Edge BDM for PPI calculated\n')
# ============================================================================================= #
# EXPORTING
genes = len(nodebdm_ppi)
memUse = ps.memory_info()
total_time=time.time()-start
output_data = {}
output_data['nodebdm_ppi'] = nodebdm_ppi
output_data['rem_edgebdm_ppi'] = rem_edgebdm_ppi
output_data['vms_ppi'] = memUse.vms
output_data['rss_ppi'] = memUse.rss
output_data['time_ppi'] = total_time
output_data['jobs_ppi'] = jobs
output_data['partition_type'] = part
path = os.getcwd()
words = input_file.split('_')
output_file = path + '/data_structures/BDM/PPI_BDM_' + words[2] + '_genes_' + str(genes)
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
print('Output data exported in ', output_file,'\n')
