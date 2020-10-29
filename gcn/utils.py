from enum import Enum

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix


class Dataset(Enum):
    Cora = 0
    CiteSeer = 1
    PubMed = 2


def load_data(dataset_name: Dataset, load_dir="planetoid"):
    dataset = Planetoid(root=load_dir, name=dataset_name.name)
    data = dataset[0]  # a single graph

    # read & normalize features
    features = data.x.clone()
    features_sum = features.sum(1).unsqueeze(1)
    features_sum[features_sum == 0] = 1.
    features = torch.div(features, features_sum)

    # read train, test, valid labels based on public splits of this data
    ignore_index = nn.CrossEntropyLoss().ignore_index  # = -100, used to ignore not allowed labels in CE loss
    num_classes = len(set(data.y.numpy()))
    labels = data.y.clone()
    train_labels = set_labels(data.y.clone(), data.train_mask, ignore_index)
    val_labels = set_labels(data.y.clone(), data.val_mask, ignore_index)
    test_labels = set_labels(data.y.clone(), data.test_mask, ignore_index)

    # read & normalize adjacency matrix
    adjacency_matrix, adj_csr = get_adjacency_matrix(data.edge_index)

    # compute rescaled laplacian
    laplacian_matrix = get_laplacian_matrix(adj_csr)

    return features, labels, train_labels, val_labels, test_labels, adjacency_matrix, laplacian_matrix, num_classes


def set_labels(initial_labels, set_mask, ignore_label):
    initial_labels[~set_mask] = ignore_label
    return initial_labels


def get_adjacency_matrix(edge_index):
    # working with scipy sparse since current PyTorch version doesn't support sparse x sparse multiplication
    adj = to_scipy_sparse_matrix(edge_index)
    adj += sparse.eye(adj.shape[0])  # add self loops
    degree_for_norm = sparse.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())  # D^(-0.5)
    adj_hat_csr = degree_for_norm.dot(adj.dot(degree_for_norm))  # D^(-0.5) * A * D^(-0.5)
    adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)
    # to torch sparse matrix
    indices = torch.from_numpy(np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_hat_coo.data)
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(adj_hat_coo.shape))

    return adjacency_matrix, adj_hat_csr


def get_laplacian_matrix(adjacency_matrix_csr: sparse.csr_matrix):
    # since adjacency_matrix_csr is already in form D^(-0.5) * A * D^(-0.5), we can simply get normalized laplacian by:
    laplacian = sparse.eye(adjacency_matrix_csr.shape[0]) - adjacency_matrix_csr
    # rescaling laplacian
    max_eigenval = sparse.linalg.eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0]
    laplacian = 2 * laplacian / max_eigenval - sparse.eye(adjacency_matrix_csr.shape[0])
    # to torch sparse matrix
    laplacian = laplacian.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((laplacian.row, laplacian.col)).astype(np.int64))
    values = torch.from_numpy(laplacian.data)
    laplacian_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(laplacian.shape))
    return laplacian_matrix
