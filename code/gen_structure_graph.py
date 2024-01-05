""" Generating structure graphs for graph convolutional neural networks """
import os
from os.path import isfile
from enum import Enum, auto

import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from biopandas.pdb import PandasPdb

import constants
import utils


class GraphType(Enum):
    LINEAR = auto()
    COMPLETE = auto()
    DISCONNECTED = auto()
    DIST_THRESH = auto()
    DIST_THRESH_SHUFFLED = auto()


def save_graph(g, fn):
    """ Saves graph to file """
    nx.write_gexf(g, fn)


def load_graph(fn):
    """ Loads graph from file """
    g = nx.read_gexf(fn, node_type=int)
    return g


def shuffle_nodes(g, seed=7):
    """ Shuffles the nodes of the given graph and returns a copy of the shuffled graph """
    # get the list of nodes in this graph
    nodes = g.nodes()

    # create a permuted list of nodes
    np.random.seed(seed)
    nodes_shuffled = np.random.permutation(nodes)

    # create a dictionary mapping from old node label to new node label
    mapping = {n: ns for n, ns in zip(nodes, nodes_shuffled)}

    g_shuffled = nx.relabel_nodes(g, mapping, copy=True)

    return g_shuffled


def linear_graph(num_residues):
    """ Creates a linear graph where each each node is connected to its sequence neighbor in order """
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, num_residues))
    for i in range(num_residues-1):
        g.add_edge(i, i+1)
    return g


def complete_graph(num_residues):
    """ Creates a graph where each node is connected to all other nodes"""
    g = nx.complete_graph(num_residues)
    return g


def disconnected_graph(num_residues):
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, num_residues))
    return g


def dist_thresh_graph(dist_mtx, threshold):
    """ Creates undirected graph based on a distance threshold """
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, dist_mtx.shape[0]))

    # loop through each residue
    for rn1 in range(len(dist_mtx)):
        # find all residues that are within threshold distance of current
        rns_within_threshold = np.where(dist_mtx[rn1] < threshold)[0]

        # add edges from current residue to those that are within threshold
        for rn2 in rns_within_threshold:
            # don't add self edges
            if rn1 != rn2:
                g.add_edge(rn1, rn2)
    return g


def ordered_adjacency_matrix(g):
    """ returns the adjacency matrix ordered by node label in increasing order as a numpy array """
    node_order = sorted(g.nodes())
    adj_mtx = nx.to_numpy_matrix(g, nodelist=node_order)
    return np.asarray(adj_mtx).astype(np.float32)


def cbeta_distance_matrix(pdb_fn, start=0, end=None):
    # note that start and end are not going by residue number
    # they are going by whatever the listing in the pdb file is

    # read the pdb file into a biopandas object
    ppdb = PandasPdb().read_pdb(pdb_fn)

    # group by residue number
    grouped = ppdb.df["ATOM"].groupby("residue_number")

    # a list of coords for the cbeta or calpha of each residue
    coords = []

    # loop through each residue and find the coordinates of cbeta
    for i, (residue_number, values) in enumerate(grouped):

        # skip residues not in the range
        end_index = (len(grouped) if end is None else end)
        if i not in range(start, end_index):
            continue

        residue_group = grouped.get_group(residue_number)

        atom_names = residue_group["atom_name"]
        if "CB" in atom_names.values:
            # print("Using CB...")
            atom_name = "CB"
        elif "CA" in atom_names.values:
            # print("Using CA...")
            atom_name = "CA"
        else:
            raise ValueError("Couldn't find CB or CA for residue {}".format(residue_number))

        # get the coordinates of cbeta (or calpha)
        coords.append(
            residue_group[residue_group["atom_name"] == atom_name][["x_coord", "y_coord", "z_coord"]].values[0])

    # stack the coords into a numpy array where each row has the x,y,z coords for a different residue
    coords = np.stack(coords)

    # compute pairwise euclidean distance between all cbetas
    dist_mtx = cdist(coords, coords, metric="euclidean")

    return dist_mtx


def gen_graph(graph_type, res_dist_mtx, dist_thresh=7, shuffle_seed=7, graph_save_dir=None, save=False):
    """ generate the specified structure graph using the specified residue distance matrix """
    if graph_type is GraphType.LINEAR:
        g = linear_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "linear.graph")

    elif graph_type is GraphType.COMPLETE:
        g = complete_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "complete.graph")

    elif graph_type is GraphType.DISCONNECTED:
        g = disconnected_graph(len(res_dist_mtx))
        save_fn = None if not save else os.path.join(graph_save_dir, "disconnected.graph")

    elif graph_type is GraphType.DIST_THRESH:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)
        save_fn = None if not save else os.path.join(graph_save_dir, "dist_thresh_{}.graph".format(dist_thresh))

    elif graph_type is GraphType.DIST_THRESH_SHUFFLED:
        g = dist_thresh_graph(res_dist_mtx, dist_thresh)
        g = shuffle_nodes(g, seed=shuffle_seed)
        save_fn = None if not save else \
            os.path.join(graph_save_dir, "dist_thresh_{}_shuffled_r{}.graph".format(dist_thresh, shuffle_seed))

    else:
        raise ValueError("Graph type {} is not implemented".format(graph_type))

    if save:
        if isfile(save_fn):
            print("err: graph already exists: {}. to overwrite, delete the existing file first".format(save_fn))
        else:
            utils.mkdir(graph_save_dir)
            save_graph(g, save_fn)

    return g


def gen_all_graphs():
    """ generate all structure graphs for all datasets """
    thresholds = [4, 5, 6, 7, 8, 9, 10]
    shuffle_seed = 7
    for ds_name in constants.DATASETS.keys():
        cbeta_mtx = cbeta_distance_matrix(constants.DATASETS[ds_name]["pdb_fn"])
        for graph_type in GraphType:
            if graph_type in [GraphType.DIST_THRESH, GraphType.DIST_THRESH_SHUFFLED]:
                for threshold in thresholds:
                    gen_graph(graph_type, cbeta_mtx, dist_thresh=threshold, shuffle_seed=shuffle_seed,
                              graph_save_dir="data/{}/graphs".format(ds_name), save=True)
            else:
                gen_graph(graph_type, cbeta_mtx, graph_save_dir="data/{}/graphs".format(ds_name), save=True)


def main():
    gen_all_graphs()


if __name__ == "__main__":
    main()
