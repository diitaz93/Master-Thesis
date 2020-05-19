#Python 3

import numpy as np
import scipy.sparse as sp
import time
import os
import psutil
import shelve
from pybdm import BDM
from pybdm.utils import decompose_dataset
from joblib import Parallel, delayed
from node import NodePerturbationExperiment

# psutil & time BEGIN
start = time.time() #in seconds
pid = os.getpid()
ps= psutil.Process(pid)

input_data = shelve.open('./results/decagon')
for key in input_data:
    globals()[key]=input_data[key]
input_data.close()
print('Input data loaded')
bdm = BDM(ndim=2)
node_per = NodePerturbationExperiment(bdm,np.array(ppi_adj.todense())
                                      ,metric='bdm',bipartite_network=False)
bdm_ppi = node_per.run()
print('BDM for PPI calculated')
node_per = NodePerturbationExperiment(bdm,np.array(dti_adj.todense())
                                      ,metric='bdm',bipartite_network=True)
bdm_drugs_dti,bdm_genes_dti = node_per.run()
print('BDM for DTI calculated')
bdm_ddi_list = []
for i in ddi_adj_list:
    node_per = NodePerturbationExperiment(bdm,np.array(i.todense())
                                          ,metric='bdm',bipartite_network=False)
    print('next ddi matrix')
    bdm_ddi_list.append(node_per.run())
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
