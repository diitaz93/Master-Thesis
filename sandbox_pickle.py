import numpy as np
import scipy.sparse as sp
import pickle

# An arbitrary collection of objects supported by pickle.
data = {
    'a': [1, 2.0, 3, 4+6j],
    'b': np.zeros([10,10]),
    'c': sp.csc_matrix(np.ones([50,50]))
}

with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, protocol=3)
