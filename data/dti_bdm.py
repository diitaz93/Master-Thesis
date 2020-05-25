#Python 3
import numpy as np
import scipy.sparse as sp
import time
import os
import psutil
import shelve
from pybdm import BDM
from algorithms import PerturbationExperiment, NodePerturbationExperiment
from getpass import getuser

# psutil & time BEGIN
start = time.time() 
pid = os.getpid()
ps= psutil.Process(pid)
with shelve.open('./data_structures/decagon') as dec:
    dti_adj = dec['dti_adj']
print('Input data loaded')
jobs = 8
usrnm = getuser()
bdm = BDM(ndim=2)
# DTI
# Node perturbation
dti_nodeper = NodePerturbationExperiment(bdm,metric='bdm',bipartite_network=True, 
                                         parallel=True,jobs=jobs)
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

genes,drugs = dti_adj.shape
memUse = ps.memory_info()
total_time=time.time()-start
filename = './data_structures/dti_bdm_genes'+str(genes)+'_drugs'+str(drugs)+'_'+usrnm+str(jobs)
output_data = shelve.open(filename,'n',protocol=2)
output_data['nodebdm_drugs_dti'] = nodebdm_drugs_dti
output_data['nodebdm_genes_dti'] = nodebdm_genes_dti
output_data['edgebdm_drugs_dti'] = edgebdm_drugs_dti
output_data['edgebdm_genes_dti'] = edgebdm_genes_dti
output_data['vms'] = memUse.vms
output_data['rss'] = memUse.rss
output_data['total_time'] = total_time
output_data['jobs'] = jobs
output_data.close()
print('Output data exported')
