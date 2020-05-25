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
"""
# ============================================================================================= #
import numpy as np
import scipy.sparse as sp
import time
import os
import psutil
import shelve
from pybdm import BDM
from algorithms import PerturbationExperiment, NodePerturbationExperiment
from getpass import getuser
# Settings and loading of adj matrix
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
with shelve.open('./data_structures/decagon') as dec:
    ppi_adj = dec['ppi_adj']
print('Input data loaded')
jobs = 8
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
output_data = shelve.open(filename,'n',protocol=2)
output_data['nodebdm_ppi'] = nodebdm_ppi
output_data['edgebdm_ppi'] = edgebdm_ppi
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
output_data['total_time'] = total_time
output_data['jobs'] = jobs
output_data.close()
print('Output data exported')
