#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# ppi_bdm.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Calculates the algoritmic complexity of the gene network of the DECAGON dataset. The dataset is
given as an adjacency matrix. The code uses the package pybdm to calculate the complexity
contribution of each node and its corresponding edges. The calculated feature vector along with 
relevant data are exported as a python shelf.

Parameters
----------
path : string
    (Relative) path to the file of data structures.
"""
# ============================================================================================= #
import numpy as np
import scipy.sparse as sp
import time
import os
import sys
import psutil
import pickle
from pybdm import BDM
from algorithms import PerturbationExperiment, NodePerturbationExperiment
from getpass import getuser
# Settings and loading of adj matrix
input_file = str(sys.argv[1])
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
with open(input_file, 'rb') as f:
    ppi_adj = pickle.load(f)['ppi_adj']
print('Input data loaded')
jobs = 48
usrnm = getuser()
bdm = BDM(ndim=2)
# ============================================================================================= #
# CALCULATION
# Node perturbation
ppi_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                         parallel=True,jobs=jobs)
ppi_nodeper.set_data(np.array(ppi_adj.todense()))
print("Initial BDM calculated for nodes")
nodebdm_ppi = ppi_nodeper.run()
print('Node BDM for PPI calculated')
# Edge perturbation
ppi_edgeper = PerturbationExperiment(bdm, bipartite_network=False)
ppi_edgeper.set_data(np.array(ppi_adj.todense()))
print("Initial BDM calculated for nodes")
edgebdm_ppi = ppi_edgeper.node_equivalent()
print('Edge BDM for PPI calculated')
# ============================================================================================= #
# EXPORTING
genes = len(nodebdm_ppi)
memUse = ps.memory_info()
total_time=time.time()-start
filename = './data_structures/ppi_bdm_genes'+str(genes)+'_'+usrnm+str(jobs)
output_data = {}
output_data['nodebdm_ppi'] = nodebdm_ppi
output_data['edgebdm_ppi'] = edgebdm_ppi
output_data['vms_ppi'] = memUse.vms
output_data['rss_ppi'] = memUse.rss
output_data['time_ppi'] = total_time
output_data['jobs_ppi'] = jobs
with open(filename, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
print('Output data exported')
