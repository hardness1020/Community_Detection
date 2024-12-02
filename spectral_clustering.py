import numbers
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def normalize_eigenvectors(e):
	"""
	Normalizes an eigenvector

	Parameters
	----------
	e: 1D np.array
		A single eigenvector

	Returns
	-------
	e: 1D np.array
		Normalized eigenvector
	"""
	return e/np.sqrt(np.sum(e**2))


def laplacian(G, laplacian_type="unnormalized"):
	"""
	Create the Laplacian from a graph

	Parameters
	----------
	G: nx.graph
		The graph for which to construct the Laplacian for
	laplacian_type: Type of laplacian
		"unnormalized": L = D - W
		"symmetric": L = I - D^{-1/2}*W*D^{-1/2}
		"random_walk": I - D^(-1)*W

	Returns
	-------
	L: np.array
		The laplacian of G
	"""

	D = np.diag(np.sum(np.array(nx.adjacency_matrix(G).todense()), axis=1))
	W = nx.adjacency_matrix(G).toarray()

	assert D.shape == W.shape, "Shapes of D and W don't match."

	L = None

	if laplacian_type == "unnormalized":
		L = D - W
	elif laplacian_type == "symmetric":
		I = np.ones(D.shape)
		D_inv_root = np.linalg.inv(np.sqrt(D))

		L = I - np.dot(D_inv_root, W).dot(D_inv_root)
	elif laplacian_type == "random_walk":
		I = np.ones(D.shape)
		D_inv = np.linalg.matrix_power(D, -1)

		L = I - np.matmul(D_inv, W)
	else:
		raise ValueError("Laplacian type can be 'normalized' or 'unnormalized'.")

	return L

def generate_labels_dict(G, kmeans):
	"""
	Creates a dictionary with keys as cluster numbers and values
	as nodes in G.
	E.g. {0: [1, 2], 1: [0], 3: [3, 4, 5]}

	Parameters
	----------
	G: nx.Graph
		Graph with the nodes in question
	kmeans: sklearn.cluster.KMeans
		A KMeans object which has already been fit on the data

	Returns
	-------
	labels_dict: Dict[int, List[int]]
		Mapping between cluster number and nodes
	"""

	num_nodes = len(G.nodes)
	num_clusters = kmeans.n_clusters

	labels_dict = {c: [] for c in range(num_clusters)}
	
	for i in range(num_nodes):
		labels_dict[kmeans.labels_[i]].append(i)

	return labels_dict


def visualize_graph(G, pos, labels_dict=None, colors=None, node_size=100, edge_alpha=0.1, labels=False):
	"""
	Visualizes graph with clusters as different colors.

	Parameters
	----------
	G: nx.graph
		Graph to visualize
	pos: nx layout (Dict)
		Layout of the graph
	labels_dict: Dict[int, List[int]]
	colors: List[str]
	node_size: int
	edge_alpha: float
	labels: bool
	"""

	# Draw nodes
	if labels_dict is not None and colors is not None:
		for k, v in labels_dict.items():

			nx.draw_networkx_nodes(G, 
				pos,
				nodelist=v,
				node_color=colors[k],
				node_size=node_size
			)
	else:
		nx.draw_networkx_nodes(G, pos, node_size=node_size) 
		
	# Add labels
	if labels:
		nx.draw_networkx_labels(G, pos)

	# Draw edges
	nx.draw_networkx_edges(G, pos, width=1.0, alpha=edge_alpha)
 
 
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _init_arpack_v0(size, random_state):
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0