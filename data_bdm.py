#Python 3

import numpy as np
import scipy.sparse as sp
import time
import os
import psutil
import shelve
from pybdm import BDM
from node import NodePerturbationExperiment
from algorithms import PerturbationExperiment

# psutil & time BEGIN
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
input_data = shelve.open('./results/decagon')
for key in input_data:
    globals()[key]=input_data[key]
input_data.close()
print('Input data loaded')
# PPI
# Node perturbation
ppi_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                      parallel=True,jobs=8)
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

# DTI
# Node perturbation
dti_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=True, 
                                      parallel=True,jobs=8)
dti_nodeper.set_data(np.array(dti_adj.todense()))
print("Initial BDM calculated for nodes")
nodebdm_genes_dti,nodebdm_drugs_dti = dti_nodeper.run()
print('BDM for DTI calculated')
# Edge perturbation
dti_edgeper = PerturbationExperiment(bdm, bipartite_network=True)
dti_edgeper.set_data(np.array(dti_adj.todense()))
print("Initial BDM calculated for nodes")
edgebdm_genes_dti, edgebdm_drugs_dti = dti_edgeper.node_equivalent()
print('Edge BDM for DTI calculated')

# DDI
nodebdm_ddi_list = []
edgebdm_ddi_list = []
ddi_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=False,
                                          parallel=True,jobs=8)
ddi_edgeper = PerturbationExperiment(bdm, bipartite_network=False)
for i in ddi_adj_list:
    ddi_nodeper.set_data(np.array(i.todense()))
    ddi_edgeper.set_data(np.array(i.todense()))
    print('set data')
    nodebdm_ddi_list.append(ddi_nodeper.run())
    edgebdm_ddi_list.append(ddi_edgeper.node_equivalent())
    print('next ddi matrix')
print('Node and Edge BDM for DDI calculated')

memUse = ps.memory_info()
total_time=time.time()-start
output_data = shelve.open('./results/bdm','n',protocol=2)
output_data['nodebdm_ppi'] = nodebdm_ppi
output_data['edgebdm_ppi'] = edgebdm_ppi
output_data['nodebdm_drugs_dti'] = nodebdm_drugs_dti
output_data['nodebdm_genes_dti'] = nodebdm_genes_dti
output_data['edgebdm_drugs_dti'] = edgebdm_drugs_dti
output_data['edgebdm_genes_dti'] = edgebdm_genes_dti
output_data['nodebdm_ddi_list'] = nodebdm_ddi_list
output_data['edgebdm_ddi_list'] = edgebdm_ddi_list
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
output_data['total_time'] = total_time
output_data.close()
print('Output data exported')
