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