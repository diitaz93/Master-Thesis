#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================================= #
# ddi_bdm.py                                                                                    #
# Author: Juan Sebastian Diaz Boada                                                             #
# Creation Date: 23/05/2020                                                                     #
# ============================================================================================= #
"""
Calculates the algoritmic complexity of the drug interaction network of the DECAGON dataset. 
The dataset is given as a list of adjacency matrices, each of dimension 𝑁𝑑𝑟𝑢𝑔𝑠×𝑁𝑑𝑟𝑢𝑔𝑠, 
corresponding to the connectivity per each joint side effect. The code uses the package pybdm 
to calculate the complexity contribution of each node and its corresponding edges per side 
effect. The result is a list of feature vectors, exported as a pickle readable format file 
along with relevant data.  

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
# Settings and loading of the list of adj matrices
input_file = str(sys.argv[1])
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
warnings.filterwarnings("ignore")
with open(input_file, 'rb') as f:
    ddi_adj_list = pickle.load(f)['ddi_adj_list']
print('\nInput data loaded\n')
jobs = 32
bdm = BDM(ndim=2, partition=PartitionRecursive)
part = 'PartitionRecursive'
# ============================================================================================= #
# CALCULATION
nodebdm_ddi_list = []
add_edgebdm_ddi_list = []
rem_edgebdm_ddi_list = []
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
    rem_edgebdm_ddi_list.append(ddi_edgeper.run_removing_edges())
    prog = count*100/total
    count += 1
    print(prog,'% completed')
print('Node and Edge BDM for DDI calculated\n')
# ============================================================================================= #
# EXPORTING
drugs = np.shape(ddi_adj_list[0])[0]
memUse = ps.memory_info()
total_time=time.time()-start
output_data = {}
output_data['nodebdm_ddi_list'] = nodebdm_ddi_list
output_data['rem_edgebdm_ddi_list'] = rem_edgebdm_ddi_list
output_data['vms_ddi'] = memUse.vms
output_data['rss_ddi'] = memUse.rss
output_data['time_ddi'] = total_time
output_data['jobs_ddi'] = jobs
output_data['partition_type'] = part
path = os.getcwd()
words = input_file.split('_')
output_file = path + '/data_structures/BDM/DDI_BDM_' + words[2] + '_se_' + str(total) +\
              '_drugs_' + str(drugs)
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f, protocol=3)
print('Output data exported in ', output_file,'\n')
