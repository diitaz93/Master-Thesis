"""Algorithms based on ``BDM`` objects."""
from itertools import product
from random import choice
import numpy as np
import warnings
from joblib import Parallel, delayed

class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM / entropy under changes
    applied to the underlying dataset. This is the main tool for detecting
    parts of a system having some causal significance as opposed
    to noise parts.

    Parts which when perturbed yield negative contribution to the overall
    complexity after change are likely to be important for the system,
    since their removal make it more noisy. On the other hand parts that yield
    positive contribution to the overall complexity after change are likely
    to be noise since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    bipartite_network: boolean
        If ``False``, the dataset (X) is the adjacency matrix of a one-type node
        network where all nodes can connect to each other, otherwise the matrix 
        represents a bipartite (two-type node) network where only pairs of nodes 
        of different types can connect, where nodes from one category are in the
        rows and nodes from the other category in the columns.

    See also
    --------
    pybdm.bdm.BDM : BDM computations

    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, PerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> pe = PerturbationExperiment(bdm, metric='bdm')
    >>> pe.set_data(X)
    >>> idx = np.argwhere(X) # Perturb only ones (1 --> 0)
    >>> delta_bdm = pe.run(idx)
    >>> len(delta_bdm) == idx.shape[0]
    True

    More examples can be found in :doc:`usage`.
    """
    def __init__(self, bdm, X=None, metric='bdm',bipartite_network=False):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self.bipartite_network = bipartite_network
        self._counter = None
        self._value = None
        self._ncounts = None
        if self.metric == 'bdm':
            self._method = self._update_bdm
        elif self.metric == 'ent':
            self._method = self._update_ent
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'ent'")
        if X is None:
            self.X = X
        else:
            self.set_data(X)

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return "<{}(metric={}) with {}>".format(cn, self.metric, bdm)

    @property
    def size(self):
        """Data size getter."""
        return self.X.size

    @property
    def shape(self):
        """Data shape getter."""
        return self.X.shape

    @property
    def ndim(self):
        """Data number of axes getter."""
        return self.X.ndim

    def set_data(self, X):
        """Set dataset for the perturbation experiment.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        if not np.isin(np.unique(X), range(self.bdm.nsymbols)).all():
            raise ValueError("'X' is malformed (too many or ill-mapped symbols)")
        self.X = X
        self._counter = self.bdm.decompose_and_count(X)
        if self.metric == 'bdm':
            self._value = self.bdm.compute_bdm(self._counter)
        elif self.metric == 'ent':
            self._value = self.bdm.compute_ent(self._counter)
            self._ncounts = sum(self._counter.values())

    def _idx_to_parts(self, idx):
        def _slice(i, k):
            start = i - i % k
            end = start + k
            return slice(start, end)
        try:
            shift = self.bdm.partition.shift
        except AttributeError:
            shift = 0
        shape = self.bdm.partition.shape
        if shift == 0:
            r_idx = tuple((k // l)*l for k, l in zip(idx, shape))
            idx = tuple(slice(k, k+l) for k, l in zip(r_idx, shape))
        else:
            idx = tuple(slice(max(0, k-l+1), k+l) for k, l in zip(idx, shape))
        yield from self.bdm.decompose(self.X[idx])

    def _update_bdm(self, idx, old_value, new_value, keep_changes):
        old_bdm = self._value
        new_bdm = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            if n > 1:
                new_bdm += np.log2((n-1) / n)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            else:
                new_bdm -= cmx
                if keep_changes:
                    del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0)
            if n > 0:
                new_bdm += np.log2((n+1) / n)
            else:
                new_bdm += cmx
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_bdm
        return new_bdm - old_bdm

    def _update_ent(self, idx, old_value, new_value, keep_changes):
        old_ent = self._value
        new_ent = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            p = n / self._ncounts
            new_ent += p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent -= p*np.log2(p)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            elif keep_changes:
                del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0) + 1
            p = n / self._ncounts
            new_ent -= p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent += p*np.log2(p)
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, value=-1, keep_changes=False):
        """Perturb element of the dataset.

        Parameters
        ----------
        idx : tuple
            Index tuple of an element.
        value : int or callable or None
            Value to assign.
            If negative then new value is randomly selected from the set
            of other possible values.
            For binary data this is just a bit flip and no random numbers
            generation is involved in the process.
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        float :
            BDM value change.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        26.91763012739709
        """
        old_value = self.X[idx]
        if value < 0:
            if self.bdm.nsymbols <= 2:
                value = 1 if old_value == 0 else 0
            else:
                value = choice([
                    x for x in range(self.bdm.nsymbols)
                    if x != old_value
                ])
        if old_value == value:
            return 0
        return self._method(idx, old_value, value, keep_changes)

    def run(self, idx=None, values=None, keep_changes=False):
        """Run perturbation experiment.

        Parameters
        ----------
        idx : array_like or None
            *Numpy* integer array providing indexes (in rows) of elements
            to perturb. If ``None`` then all elements are perturbed.
        values : array_like or None
            Value to assign during perturbation.
            Negative values correspond to changing value to other
            randomly selected symbols from the alphabet.
            If ``None`` then all values are assigned this way.
            If set then its dimensions must agree with the dimensions
            of ``idx`` (they are horizontally stacked).
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        array_like
            1D float array with perturbation values.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> changes = np.array([10, 20])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        if idx is None:
            indexes = [ range(k) for k in self.X.shape ]
            idx = np.array([ x for x in product(*indexes) ], dtype=int)
        if values is None:
            values = np.full((idx.shape[0], ), -1, dtype=int)
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=np.column_stack((idx, values))
        )

    def node_equivalent(self):
        """Returns the average complexity value of the edges for each node.

        Returns
        -------
        array or tuple of arrays :
            If bipartite_network=True, it returns the value for each node of the corresponding
            set, being the row nodes the ones of the first array and the column nodes of the
            second. If bipartite_network=False, the matrix is symmetric and therefore the 
            row and column nodes are the same, returning only one array.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=2)
        >>> b = np.random.randint(0,2,(10,10))
        >>> X = ((b + b.T)/2).astype(int)
        >>> perturbation = PerturbationExperiment(bdm, X, bipartite_network=False)
        >>> perturbation.node_equivalent()
        array([-0.96990006,  0.51649434, -0.37643584,  0.70535874, -0.16374676,
        1.02257465, -0.32975466,  0.63645683,  0.        ,  0.        ])
        """
        pert = self.run().reshape(self.shape)
        if self.bipartite_network:
            row_nodes = np.mean(pert,axis=1)
            col_nodes = np.mean(pert,axis=0)
            return row_nodes,col_nodes
        else:
            return np.mean(pert,axis=0)
            
        
class NodePerturbationExperiment:
    """Node Perturbation experiment class.

    Node perturbation experiment studies change of BDM / entropy of a 
    network (given their adjacency matrix) when existing nodes are removed.
    This is the main tool for detecting nodes of a network having some 
    causal significance as opposed to noise parts.

    Nodes which when removed yield negative contribution to the overall
    complexity  are likely to be important for the system, since their 
    removal make it more noisy. On the other hand nodes that yield positive 
    contribution to the overall complexity after removal are likely to be noise 
    since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    bipartite_network: boolean
        If ``False``, the dataset (X) is the adjacency matrix of a one-type node
        network where all nodes can connect to each other, otherwise the matrix 
        represents a bipartite (two-type node) network where only pairs of nodes 
        of different types can connect, where nodes from one category are in the
        rows and nodes from the other category in the columns.
    parallel: boolean
        Determines wether the bdm calculations are done in parallel
    jobs: int
        If parallel=True, is the number of jobs in the parallel computation

    See also
    --------
    pybdm.bdm.BDM : BDM computations

    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, NodePerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> npe = NodePerturbationExperiment(bdm, metric='bdm')
    >>> npe.set_data(X)

    >>> idx = [1,5,10,50] # Remove these specific nodes
    >>> delta_bdm = npe.run(idx)
    >>> len(delta_bdm) == len(idx)
    True

    More examples can be found in :doc:`usage`.
    """
    def __init__(self, bdm, X=None, metric='bdm', bipartite_network=False, parallel=False, jobs=1):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self.bipartite_network = bipartite_network
        self.parallel = parallel
        self.jobs = jobs
        self._value = None
        if self.metric == 'bdm':
            self._method = self._update_bdm
        elif self.metric == 'ent':
            self._method = self._update_ent
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'ent'")
        if X is None:
            self.X = X
        else:
            self.set_data(X)

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return "<{}(metric={}) with {}>".format(cn, self.metric, bdm)

    @property
    def size(self):
        """Data size getter."""
        return self.X.size

    @property
    def shape(self):
        """Data shape getter."""
        return self.X.shape

    def set_data(self, X):
        """Set dataset for the perturbation experiment.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        if not np.isin(np.unique(X), range(self.bdm.nsymbols)).all():
            raise ValueError("'X' is malformed (too many or ill-mapped symbols)")
        self.X = X
        if not self.bipartite_network and self.shape[0]!=self.shape[1]:
            raise ValueError("'X' has to be a squared matrix for non-bipartite network")
        if self.metric == 'bdm':
            self._value = self.bdm.bdm(self.X)
        elif self.metric == 'ent':
            self._value = self.bdm.ent(self.X)   

    def _update_bdm(self, idx, axis, keep_changes):
        old_bdm = self._value
        if not self.bipartite_network:
            newX = np.delete(self.X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(self.X,idx,axis=axis)
        new_bdm = self.bdm.bdm(newX)
        if keep_changes:
            self.X = newX
            self._value = new_bdm
        return new_bdm - old_bdm

    def _update_ent(self, idx, axis, keep_changes):
        old_ent = self._value
        if not self.bipartite_network:
            newX = np.delete(self.X,idx,axis=0)
            newX = np.delete(newX,idx,axis=1)
        else:
            newX = np.delete(self.X,idx,axis=axis)
        new_ent = self.bdm.ent(newX)
        if keep_changes:
            self.X = newX
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, axis=0, keep_changes=False):
        """Delete node of the dataset.

        Parameters
        ----------
        idx : int
            Index of row or column of node in the dataset.
        axis : int
            Axis of adjaceny matrix.
        keep_changes : bool
            If ``True`` the deletion of the node in the dataset is preserved.

        Returns
        -------
        float :
            BDM value change.

        Examples
        --------
        >>> import numpy as np
        >>> from pybdm import BDM, NodePerturbationExperiment
        >>> bdm = BDM(ndim=2)
        >>> X = np.random.randint(0, 2, (100, 100))
        >>> perturbation = NodePerturbationExperiment(bdm, X)
        >>> perturbation.perturb(7)
        -1541.9845807106612
        """
        
        return self._method(idx, axis, keep_changes)

    def run(self, idx1=None, idx2=None):
        """Run node perturbation experiment. Calls the function self.perturb for
        each node index and keep_changes=False.

        Parameters
        ----------
        idx1 : a list of node indices to be perturbed in the adyacency matrix.
            If the network is bipartite, their entries are the row-indices of the nodes
            to be removed, otherwise is the index for both rows and columns. If is None
            all the nodes are perturbed, if is empty ([]), no nodes are perturbed.
        idx2 : a list of node indices to be perturbed in the adyacency matrix.
            If the network is bipartite, their entries are the column-indices of the nodes
            to be removed, otherwise is ignored. If is None all the nodes are perturbed, 
            if is empty ([]), no nodes are perturbed.

        Returns
        -------
        For bipartite networks with indices (or None) in both axes: 
            Tuple of 1D arrays with perturbation values corresponding to row and column
            indices of nodes.
        For non-bipartite networks or only one-index-array networks:
            One 1D float arrays with perturbation values corresponding to nodes in 
            adjacency matrix.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=2)
        >>> X = np.random.randint(0,2,size=(100, 100))
        >>> idx1 = [1,6,9,20,56,70]
        >>> perturbation = NodePerturbationExperiment(bdm, X)
        >>> data = perturbation.run(idx)
        >>> print(type(data),len(data))
        <class 'numpy.ndarray'> 6
        """
        if not self.bipartite_network:
            if idx1 == []:
                raise ValueError("idx1 can not be empty for bipartire_network=False")
            if idx2 is not None:
                warnings.warn("Indices in idx2 ignored, changing only indices in idx1")
            if idx1 is None:
                idx1 = np.arange(self.shape[0])
            if self.parallel:
                output = np.array(Parallel(n_jobs=self.jobs)(delayed(self._method)
                                                    (x,axis=0,keep_changes=False) for x in idx1))
            else:
                output = np.array([self._method(x,axis=0, keep_changes=False)for x in idx1])
            return output
        else:
            if idx1 == [] and idx2 == []:
                raise ValueError("There has to be indices to change in either idx1 or idx2")
            if idx1 is None:
                idx1 = np.arange(self.shape[0])
            if idx2 is None:
                idx2 = np.arange(self.shape[1])
            if idx1 != []:
                if self.parallel:
                    out_rows = np.array(Parallel(n_jobs=self.jobs)(delayed(self._method)
                                                    (x,axis=0,keep_changes=False) for x in idx1))
                else:
                    out_rows = np.array([self._method(x, axis=0,keep_changes=False)for x in idx1])
                if idx2 == []:
                    return out_rows
            if idx2 != []:
                if self.parallel:
                    out_cols = np.array(Parallel(n_jobs=self.jobs)
                                        (delayed(self._method)(x,axis=1,keep_changes=False)
                                         for x in idx2))
                else:
                    out_cols = np.array([self._method(x, axis=1,keep_changes=False)for x in idx2])
                if idx1 == []:
                    return out_cols
            return out_rows, out_cols
