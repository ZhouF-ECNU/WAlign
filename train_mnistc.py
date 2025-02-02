import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import csv
from itertools import product
from tqdm import tqdm
from modeling.net import SemiADNet
from utils import aucPerformance
from modeling.layers import build_criterion
from datasets.mnistc3_SpectralClustering import MNISTC_Dataset
from torch.utils.data import DataLoader
from dataloaders.utlis_cluster3 import BalancedBatchSampler
from sklearn.metrics import auc, precision_recall_curve
import torch.nn.functional as F
import random

torch.set_printoptions(threshold=10000)
torch.cuda.set_device(0)

np.set_printoptions(threshold=np.inf)

def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)

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

def mmd_loss(x, y, kernel_bandwidth=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) loss between two distributions."""
    xx_kernel = gaussian_kernel(x, x, kernel_bandwidth)
    yy_kernel = gaussian_kernel(y, y, kernel_bandwidth)
    xy_kernel = gaussian_kernel(x, y, kernel_bandwidth)

    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def mmd_loss_with_weight(x, y, w_xx, w_yy, w_xy, kernel_bandwidth=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) loss between two distributions."""
    xx_kernel = gaussian_kernel(x, x, kernel_bandwidth)
    yy_kernel = gaussian_kernel(y, y, kernel_bandwidth)
    xy_kernel = gaussian_kernel(x, y, kernel_bandwidth)
    xx_kernel = xx_kernel * w_xx
    yy_kernel = yy_kernel * w_yy
    xy_kernel = xy_kernel * w_xy

    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

class Trainer(object):

    def __init__(self, args, train_set, test_set):
        self.args = args
        self.train_set = train_set
        self.test_set = test_set

        idx = self.train_set.cluster_targets
        normal_1_idx = np.where(np.isin(idx, [1]))[0]
        normal_2_idx = np.where(np.isin(idx, [2]))[0]
        normal_3_idx = np.where(np.isin(idx, [3]))[0]
        outlier_idx = np.where(np.isin(idx, [-1]))[0]

        self.train_loader = DataLoader(dataset=self.train_set, worker_init_fn=worker_init_fn_seed, 
                                       batch_sampler=BalancedBatchSampler(args, self.train_set, normal_1_idx, normal_2_idx, normal_3_idx, outlier_idx))
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=args.batch_size)

        self.model = SemiADNet(args)
        self.criterion = build_criterion(args.criterion)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        for i, sample in enumerate(self.train_loader):
            input, target, semi_target, cluster_target = sample
            input, target, semi_target, cluster_target = input.float(), target.float(), semi_target.float(), cluster_target.float()
            if self.args.cuda:
                input, target, semi_target, cluster_target = input.cuda(), target.cuda(), semi_target.cuda(), cluster_target.float()

            output, feature, _ = self.model(input)
            loss = self.criterion(output, target.unsqueeze(1).float())

            batch_size = feature.size(0)
            feature = feature.view(batch_size, -1)
            n_normal = batch_size // 4
            normal_1_features = feature[:n_normal]
            normal_2_features = feature[n_normal:n_normal*2]
            normal_3_features = feature[n_normal*2:n_normal*3]
            normal_1_features = F.normalize(normal_1_features, p=2, dim=1)
            normal_2_features = F.normalize(normal_2_features, p=2, dim=1)
            normal_3_features = F.normalize(normal_3_features, p=2, dim=1)

            output_normal = output[:n_normal*3]
            # 取出当前样本状态是0还是1，-1≤sigma≤1为0，otherwise为1，标记异常为1
            state = ((output_normal < -1) | (output_normal > 1)).int()
            state = torch.cat((state, torch.ones(n_normal, 1, dtype=torch.int, device='cuda:0')), dim=0)

            # 找到feature在batch中的top10近邻，取出其被预测为1的概率(将近邻样本的state求和/选取近邻样本个数)
            distances = torch.cdist(feature, feature)
            _, nearest_indices = torch.topk(distances[:n_normal*3], k=6, largest=False, dim=1)
            nearest_indices = nearest_indices[:, 1:]
            nearest_states = state[nearest_indices]
            probabilities = nearest_states.sum(dim=1).float() / 5

            # 计算不一致分数，排序找到不一致程度高的样本(例如10个)，和真实误判作比较，看是否能够找出
            state_labels = state[:n_normal*3].float()
            sample_losses = F.binary_cross_entropy(probabilities, state_labels, reduction='none')
            _, top10_indices = torch.topk(sample_losses.squeeze(), k=1, largest=True)


            output_normal = output[:n_normal*3]
            output_normal_max = output_normal.max()
            output_normal_min = output_normal.min()
            normalized_output = (output_normal - output_normal_min) / (output_normal_max - output_normal_min)
            sigma = (2.0 - output_normal_min) / (output_normal_max - output_normal_min)

            score = normalized_output
            abs_score = torch.abs(score)
            weights = torch.where(abs_score <= sigma, torch.ones_like(score), 2 / (1 + torch.exp(abs_score - sigma)))

            a = 2
            mu = 0
            sigma_ = 1
            s_selected = output[top10_indices]
            p_selected = probabilities[top10_indices]
            abs_s_mu = torch.abs(s_selected - mu)
            term1 = torch.minimum(torch.tensor(a, dtype=torch.float32), abs_s_mu / sigma_)
            term2 = a * (1 - p_selected)
            corrected_weights = (term1 + term2) / 2

            weights[top10_indices] = corrected_weights
            
            weights_normal_1 = weights[:n_normal]
            weights_normal_2 = weights[n_normal:n_normal*2]
            weights_normal_3 = weights[n_normal*2:n_normal*3]

            normal_features = [normal_1_features, normal_2_features, normal_3_features]
            weights_normal = [weights_normal_1, weights_normal_2, weights_normal_3]
            mmd_total = 0.0

            for idx, current_features in enumerate(normal_features):
                other_indices = [i for i in range(3) if i!= idx]
                selected_idx = random.choice(other_indices)
                selected_features = normal_features[selected_idx]
                W_current = weights_normal[idx] * weights_normal[idx].T
                W_selected = weights_normal[selected_idx] * weights_normal[selected_idx].T
                W_curr_sele = weights_normal[idx] * weights_normal[selected_idx].T

                mmd = mmd_loss_with_weight(current_features, selected_features, W_current, W_selected, W_curr_sele, kernel_bandwidth=1.0)
                mmd = 50*mmd

                mmd_total += mmd

            loss += mmd_total

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            if i == 0:
                print('Epoch:%d, Batch:%d, Train loss: %.6f' % (epoch, i, train_loss / (i + 1)))

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            input, target, semi_target, cluster_target = sample
            input, target, semi_target, cluster_target = input.float(), target.float(), semi_target.float(), cluster_target.float()
            if self.args.cuda:
                input, target, semi_target, cluster_target = input.cuda(), target.cuda(), semi_target.cuda(), cluster_target.float()

            with torch.no_grad():
                output, _, _ = self.model(input)
            loss = self.criterion(output, target.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
        
        roc, pr = aucPerformance(total_pred, total_target)
        precision, recall, threshold = precision_recall_curve(total_target, total_pred)
        test_auc_pr = auc(recall, precision)
        print('Test AUC: {:.4f} | Test PRC: {:.4f}'.format(roc, test_auc_pr))
        return roc, pr

def update_results_file(file_path, header, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=60, help="the number of epochs")
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--experiment_dir', type=str, default='./normal-distribution-alignment/experiment', help="experiment dir root")
    parser.add_argument('--img_size', type=int, default=224, help="the image size of input") # mnistc input size
    parser.add_argument("--n_anomaly", type=int, default=200, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=1.0, help="the k percentage of instances in the topk module")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--batch_size", type=int, default=64, help="batch size used in SGD")
    parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
    parser.add_argument("--lr", type=float, default=0.000005)
    parser.add_argument("--weight_decay", type=float, default=0.000001)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    mnistc_dataset = MNISTC_Dataset(root='data/mnistc', random_seed=args.random_seed)
    train_set = mnistc_dataset.get_train_set()
    train_set.cluster_targets[:200] = [-1] * 200
    test_set = mnistc_dataset.get_test_set()

    results_header = ['epochs', 'batch_size', 'lr', 'weight_decay', 'AUROC', 'AUPRC']

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

    torch.manual_seed(args.random_seed)

    trainer = Trainer(args, train_set, test_set)

    for epoch in range(0, trainer.args.epochs):
        trainer.train(epoch)

    auroc, auprc = trainer.eval()

    result_data = [epochs, batch_size, lr, weight_decay, round(auroc, 4), round(auprc, 4)]
    update_results_file('results/mnistc/results.csv', results_header, result_data)
