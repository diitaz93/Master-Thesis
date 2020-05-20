#Python 3

import numpy as np
import scipy.sparse as sp
import time
import os
import psutil
import shelve
from pybdm import BDM
from node import NodePerturbationExperiment

# psutil & time BEGIN
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)

input_data = shelve.open('./results/decagon')
for key in input_data:
    globals()[key]=input_data[key]
input_data.close()
print('Input data loaded')

bdm = BDM(ndim=2)
ppi_perturbation = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                              parallel=True,jobs=16)
ppi_perturbation.set_data(np.array(ppi_adj.todense()))
print("Initial BDM calculated")
bdm_ppi = ppi_perturbation.run()
print('BDM for PPI calculated')

dti_perturbation = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=True, 
                                              parallel=True,jobs=16)
dti_perturbation.set_data(np.array(dti_adj.todense()))
print("Initial BDM calculated")
bdm_drugs_dti,bdm_genes_dti = dti_perturbation.run()
print('BDM for DTI calculated')

bdm_ddi_list = []
ddi_perturbation = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                              parallel=True,jobs=16)
for i in ddi_adj_list:
    dti_perturbation.set_data(np.array(i.todense()))
    print('set data')
    bdm_ddi_list.append(dti_perturbation.run())
    print('next ddi matrix')
print('BDM for DDI calculated')

memUse = ps.memory_info()
total_time=time.time()-start
print('Time and memory calculated')
output_data = shelve.open('./results/bdm','n',protocol=2)
output_data['bdm_ppi'] = bdm_ppi
output_data['bdm_drugs_dti'] = bdm_drugs_dti
output_data['bdm_genes_dti'] = bdm_genes_dti
output_data['bdm_ddi_list'] = bdm_ddi_list
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
output_data['total_time'] = total_time
output_data.close()
print('Output data exported')
