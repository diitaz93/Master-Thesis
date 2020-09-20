import numpy as np
import scipy.sparse as sp


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc


def sparse_to_tuple(sparse_mx):
    """ Expreses a sparse matrix given as parameter (in csr or coo form from scipy.sparse) 
        as a tuple of arrays. The first is a 2d array with the coordinates (with row and column)
        of the non-zero elements. The second output is an array with the non zero values corresponding 
        to the coordinates of first array. The third output is the shape of the dense matrix.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape