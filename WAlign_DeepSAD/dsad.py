# -*- coding: utf-8 -*-
from base_model import BaseDeepAD
from base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
torch.set_printoptions(threshold=float('Inf'))

def gaussian_kernel(x, y, kernel_bandwidth=1.0):
    """Compute the Gaussian kernel between two sets of data points."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input = ((tiled_x - tiled_y) ** 2).sum(2) / (2.0 * kernel_bandwidth ** 2.0)
    return torch.exp(-kernel_input)

def mmd_loss_with_weight(x, y, w_xx, w_yy, w_xy, kernel_bandwidth=1.0):
    """mmd with weight"""
    xx_kernel = gaussian_kernel(x, x, kernel_bandwidth)
    yy_kernel = gaussian_kernel(y, y, kernel_bandwidth)
    xy_kernel = gaussian_kernel(x, y, kernel_bandwidth)
    xx_kernel = xx_kernel * w_xx
    yy_kernel = yy_kernel * w_yy
    xy_kernel = xy_kernel * w_xy

    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


class DeepSAD(BaseDeepAD):
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=256, hidden_dims='1024,512', act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSAD, self).__init__(
            data_type='tabular', model_name='DeepSAD',
            epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.c = None
        self.sample_weights = None
        self.R = None
        self.r = 2000
        self.k = 21
        self.a = 1.5

        return

    def _compute_epoch_weights(self, net):
        """weight update for each epoch"""
        net.eval()
        dist_list = []
        z_list = []

        with torch.no_grad():
            dataset = TensorDataset(torch.from_numpy(self.train_data).float())
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for x in loader:
                x = x[0].to(self.device)
                z = net(x)
                dist = torch.sqrt(torch.sum((z - self.c) ** 2, dim=1))
                z_list.append(z.cpu())
                dist_list.append(dist.cpu())
        
        num_normal = np.sum(self.train_label == 0)
        num_abnormal = np.sum(self.train_label == 1)
        z_all = torch.cat(z_list)
        dist_all = torch.cat(dist_list)
        dist_remove_anomaly = dist_all[:num_normal]
        self.R = torch.quantile(dist_remove_anomaly, 0.95).item()

        state = torch.zeros_like(dist_all, dtype=torch.int)
        state[:num_normal] = (dist_all[:num_normal] > self.R).int()
        state[num_normal:] = 1

        z_distances = torch.cdist(z_all, z_all)
        _, nearest_indices = torch.topk(z_distances[:num_normal],k=self.k, largest=False, dim=1)
        nearest_indices = nearest_indices[:, 1:]
        nearest_states = state[nearest_indices]
        probabilities = nearest_states.float().sum(dim=1) / 20

        state_normal = state[:num_normal].float()
        sample_losses = F.binary_cross_entropy(probabilities, state_normal, reduction='none')
        _, topr_indices = torch.topk(sample_losses, k=self.r, largest=True)
        
        weights = torch.where(
            dist_all <= self.R,
            torch.ones_like(dist_all),
            2.0 / (1.0 + torch.exp(dist_all - self.R)))
        
        score_topr = dist_remove_anomaly[topr_indices]
        p_topr = probabilities[topr_indices]
        term1 = torch.minimum(torch.tensor(self.a, dtype=torch.float32), score_topr / self.R)
        term2 = self.a * (1-p_topr)
        corrected_weights = (term1 + term2) / 2
        weights[topr_indices] = corrected_weights

        self.sample_weights = weights.numpy()

    def training_prepare(self, X, y, cluster_labels):
        known_anom_id = np.where(y == 1)
        y = np.zeros_like(y)
        y[known_anom_id] = -1

        counter = Counter(y)
        normal_indices = np.where(y == 0)[0]
        cluster_0_count = np.sum(cluster_labels[normal_indices] == 0)
        cluster_1_count = np.sum(cluster_labels[normal_indices] == 1)

        initial_weights = np.ones(len(X))
        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long(),
                                torch.from_numpy(cluster_labels).long(),
                                torch.from_numpy(initial_weights).float())
        
        weight_map = {
            (0, 0): 1. / cluster_0_count if cluster_0_count > 0 else 0,
            (0, 1): 1. / cluster_1_count if cluster_1_count > 0 else 0,
            -1: 1. / counter[-1]
        }

        weights = [weight_map[(label.item(), cluster.item())] if label.item() == 0 
                else weight_map[label.item()]
                for _, label, cluster, _ in dataset]

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                sampler=sampler,
                                shuffle=True if sampler is None else False)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)

        self.c = self._set_c(net, train_loader)
        criterion = DSADLoss(c=self.c)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'
        return test_loader 

    def training_forward(self, batch_x, net, criterion):
        batch_x, batch_y, batch_cluster_labels, batch_weights = batch_x
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.long().to(self.device)
        batch_cluster_labels = batch_cluster_labels.long().to(self.device)
        batch_weights = batch_weights.float().to(self.device)  

        z = net(batch_x)
        loss = criterion(z, batch_y)

        indices_0 = batch_cluster_labels == 0
        indices_1 = batch_cluster_labels == 1
        z_0 = z[indices_0]
        z_1 = z[indices_1]
        weights_z_0 = batch_weights[indices_0]
        weights_z_1 = batch_weights[indices_1]
        z_0 = F.normalize(z_0, p=2, dim=1)
        z_1 = F.normalize(z_1, p=2, dim=1)

        w_xx_ = weights_z_0.unsqueeze(1) @ weights_z_0.unsqueeze(0)
        w_yy_ = weights_z_1.unsqueeze(1) @ weights_z_1.unsqueeze(0)
        w_xy_ = weights_z_0.unsqueeze(1) @ weights_z_1.unsqueeze(0)
        mmd = mmd_loss_with_weight(z_0, z_1, w_xx_, w_yy_, w_xy_, kernel_bandwidth=1.0)
        mmd = 100*mmd
        loss += mmd
        
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = criterion(batch_z)
        return batch_z, s

    def _set_c(self, net, dataloader, eps=0.1):
        net.eval()
        z_ = []
        with torch.no_grad():
            for x, _, _, _ in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSADLoss(torch.nn.Module):
    def __init__(self, c, eta=1.0, eps=1e-6, reduction='mean'):
        super(DSADLoss, self).__init__()
        self.c = c
        self.reduction = reduction
        self.eta = eta
        self.eps = eps

    def forward(self, rep, semi_targets=None, reduction=None):
        dist = torch.sum((rep - self.c) ** 2, dim=1)

        if semi_targets is not None:
            loss = torch.where(semi_targets == 0, dist,
                               self.eta * ((dist+self.eps) ** semi_targets.float()))
        else:
            loss = dist

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
