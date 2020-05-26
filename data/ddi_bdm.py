#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# ddi_bdm.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Calculates the algoritmic complexity of the drug interaction network of the DECAGON dataset. The 
dataset is given as a list of adjacency matrices corresponding to the connectivity per each joint
side effect. The code uses the package pybdm to calculate the complexity contribution of each 
node and its corresponding edges per side effect. The result is a list of feature vectors, 
exported as a python shelf.  

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
# Settings and loading of the list of adj matrices
input_file = str(sys.argv[1])
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
with open(input_file, 'rb') as f:
    ddi_adj_list = pickle.load(f)['ddi_adj_list']
print('Input data loaded')
jobs = 8
usrnm = getuser()
bdm = BDM(ndim=2)
# ============================================================================================= #
# CALCULATION
nodebdm_ddi_list = []
edgebdm_ddi_list = []
ddi_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                          parallel=True,jobs=jobs)
ddi_edgeper = PerturbationExperiment(bdm, bipartite_network=False)
total = len(ddi_adj_list)
count=1
for i in ddi_adj_list:
    ddi_nodeper.set_data(np.array(i.todense()))
    ddi_edgeper.set_data(np.array(i.todense()))
    print('set data')
    nodebdm_ddi_list.append(ddi_nodeper.run())
    edgebdm_ddi_list.append(ddi_edgeper.node_equivalent())
    prog = count*100/total
    count += 1
    print(prog,'% completed')
print('Node and Edge BDM for DDI calculated')
# ============================================================================================= #
# EXPORTING
drugs = np.shape(ddi_adj_list[0])[0]
memUse = ps.memory_info()
total_time=time.time()-start
filename = './data_structures/ddi_bdm_se'+str(total)+'_drugs'+str(drugs)+'_'+usrnm+str(jobs)
output_data = {}
output_data['nodebdm_ddi_list'] = nodebdm_ddi_list
output_data['edgebdm_ddi_list'] = edgebdm_ddi_list
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
output_data['total_time'] = total_time
output_data['jobs'] = jobs
with open(filename, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
print('Output data exported')
