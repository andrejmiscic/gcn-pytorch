import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    LAYERS: GCNConv and ChebNetConv
"""


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency_hat, x)
        return x


class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x

        if self.K > 1:
            x1 = torch.sparse.mm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2

        cheb_x = cheb_x.reshape([x.shape[0], -1])
        return cheb_x


"""
    MODELS
"""


class TwoLayerGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(TwoLayerGCN, self).__init__()

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = self.dropout(x)
        x = self.conv1(x, adjacency_hat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adjacency_hat)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, dropout=0.1, residual=False):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.residual = residual

        self.input_conv = GCNConv(input_size, hidden_size)
        self.hidden_convs = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output_conv = GCNConv(hidden_size, output_size)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_conv(x, adjacency_hat))
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, adjacency_hat)) + x
            else:
                x = F.relu(conv(x, adjacency_hat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, adjacency_hat)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class TwoLayerChebNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, k=2):
        super(TwoLayerChebNet, self).__init__()

        self.conv1 = ChebNetConv(input_size, hidden_size, k)
        self.conv2 = ChebNetConv(hidden_size, output_size, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = self.dropout(x)
        x = self.conv1(x, laplacian)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, laplacian)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class ChebNetGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, dropout=0.1, residual=False, k=2):
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout
        self.residual = residual

        self.input_conv = ChebNetConv(input_size, hidden_size, k)
        self.hidden_convs = nn.ModuleList([ChebNetConv(hidden_size, hidden_size, k) for _ in range(num_hidden_layers)])
        self.output_conv = ChebNetConv(hidden_size, output_size, k)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor, labels: torch.Tensor = None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_conv(x, laplacian))
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, laplacian)) + x
            else:
                x = F.relu(conv(x, laplacian))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, laplacian)

        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss
