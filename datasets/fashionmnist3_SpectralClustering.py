import os
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import SpectralClustering
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_printoptions(threshold=float('Inf'))
np.set_printoptions(threshold=np.inf)

class MyFashionMNIST(FashionMNIST):
    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)
        self.cluster_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        img, target, cluster_target = self.data[index], int(self.targets[index]), int(self.cluster_targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, cluster_target

class MyDataset(Dataset):
    def __init__(self, x, y, cluster_targets=None, transform=None, test_transform=None, target_transform=None):
        self.data = x
        self.labels = y
        self.cluster_targets = cluster_targets if cluster_targets is not None else torch.zeros(len(y), dtype=torch.int64)
        self.transform = transform
        self.test_transform = test_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        cluster_target = self.cluster_targets[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, cluster_target

# LeNet Encoder
class LeNetEncoder(nn.Module):
    def __init__(self):
        super(LeNetEncoder, self).__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

        self.bn99 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(self.bn99(x))
        return x


# LeNet Decoder
class LeNetDecoder(nn.Module):
    def __init__(self):
        super(LeNetDecoder, self).__init__()

        self.rep_dim = 64

        self.fc3 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(8, 32, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(128 / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

# LeNet Autoencoder
class LeNetAutoencoder(nn.Module):
    def __init__(self):
        super(LeNetAutoencoder, self).__init__()
        self.encoder = LeNetEncoder()
        self.decoder = LeNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FashionMNIST_Dataset:
    def __init__(self, root: str, model, device, random_seed: int = 42, pretrain: bool = True):
        self.root = root
        self.random_seed = random_seed
        self.model = model
        self.device = device

        if pretrain:
            self.pretrain_autoencoder()

        # 定义图像变换 transform
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.RandomRotation(180),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        target_transform = transforms.Lambda(lambda x: int(x in [5, 7, 9, 1]))

        train_set = MyFashionMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform, download=True)
        test_set = MyFashionMNIST(root=self.root, train=False, transform=test_transform, target_transform=target_transform, download=True)

        np.random.seed(self.random_seed)

        train_indices = {'labeled': {5: 100, 7: 100}, 'unlabeled': {0: 3166, 3: 3167, 4: 3167, 5: 250, 7: 250}}
        test_indices = {'labeled': {0: 1000, 3: 1000, 4: 1000, 6: 1000, 5: 100, 7: 100, 9: 100, 1: 100}}

        self.train_set, self.test_set = self.create_datasets(train_set, test_set, train_indices, test_indices)

    def pretrain_autoencoder(self, num_epochs=5):
        # 预训练Autoencoder
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        transform = transforms.Compose([transforms.ToTensor()])
        train_set_ = MyFashionMNIST(root=self.root, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_set_, batch_size=64, shuffle=True)

        # 训练模型
        print("Starting pretraining...")
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, _, _ in train_loader:
                images = images.to(self.device)
                outputs = self.model(images)     
                loss = criterion(outputs, images)
                loss = torch.mean(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        print("Pretraining completed.")

    def create_datasets(self, train_set, test_set, train_indices, test_indices):
        def get_indices(dataset, class_indices):
            indices = []
            for class_label, num_samples in class_indices.items():
                class_indices = np.where(dataset.targets == class_label)[0]
                sampled_indices = np.random.choice(class_indices, num_samples, replace=False)
                indices.append(sampled_indices)
            return np.concatenate(indices)

        labeled_train_indices = get_indices(train_set, train_indices['labeled'])
        unlabeled_train_indices = get_indices(train_set, train_indices['unlabeled'])
        labeled_test_indices = get_indices(test_set, test_indices['labeled'])

        x_train_unlabeled = train_set.data[unlabeled_train_indices]
        y_train_unlabeled = train_set.targets[unlabeled_train_indices]

        cluster_targets_labeled = torch.zeros(len(labeled_train_indices), dtype=torch.int64)

        transform_ = transforms.Compose([transforms.ToTensor()])
        feature_train_loader = DataLoader(MyDataset(x_train_unlabeled, y_train_unlabeled, None, transform=transform_), batch_size=64, shuffle=False)
        unlabeled_features = self.extract_features(feature_train_loader)

        # 计算无标签训练数据表征的中心
        center = np.mean(unlabeled_features, axis=0)

        # 筛选无标签数据
        x_train_unlabeled, y_train_unlabeled = self.filter_outlier(x_train_unlabeled, y_train_unlabeled, center)

        # 聚类无标签数据
        feature_unlabeled_loader = DataLoader(MyDataset(x_train_unlabeled, y_train_unlabeled, None, transform=transform_), batch_size=64, shuffle=False)
        train_features = self.extract_features(feature_unlabeled_loader)

        spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
        spectral.fit(train_features)
        cluster_targets_unlabeled = torch.tensor(spectral.labels_, dtype=torch.int64) + 1

        x_train = train_set.data[np.concatenate((labeled_train_indices, unlabeled_train_indices))]
        y_train = train_set.targets[np.concatenate((labeled_train_indices, unlabeled_train_indices))]
        cluster_targets = torch.cat([cluster_targets_labeled, cluster_targets_unlabeled])

        x_test = test_set.data[labeled_test_indices]
        y_test = test_set.targets[labeled_test_indices]

        train_dataset = MyDataset(x_train, y_train, cluster_targets, transform=train_set.transform, target_transform=train_set.target_transform)

        # 控制test是all还是unseen
        # filtered_indices = np.where(~np.isin(y_test.numpy(), [5,7]))[0]
        # x_test = x_test[filtered_indices]
        # y_test = y_test[filtered_indices]
        test_dataset = MyDataset(x_test, y_test, torch.zeros_like(y_test), transform=test_set.transform, target_transform=test_set.target_transform)

        return train_dataset, test_dataset

    def filter_outlier(self, x_unlabeled, y_unlabeled, center, threshold=0.05):
        transform_ = transforms.Compose([transforms.ToTensor()])
        feature_unlabeled_loader = DataLoader(MyDataset(x_unlabeled, y_unlabeled, None, transform=transform_), batch_size=64, shuffle=False)
        
        # 提取无标签数据的特征
        features_unlabeled = self.extract_features(feature_unlabeled_loader)

        # 计算到中心的距离
        distances = np.linalg.norm(features_unlabeled - center, axis=1)
        print(distances)

        threshold_distance = np.percentile(distances, 95)

        mask = distances <= threshold_distance
        filtered_x = x_unlabeled[mask]
        filtered_y = y_unlabeled[mask]

        # 找到被筛除的索引
        excluded_indices = np.where(~mask)[0]  # 反向索引
        print("Excluded indices from unlabeled data:", excluded_indices)

        # 打印被删除样本的距离值
        excluded_distances = distances[excluded_indices]
        print("Distances of excluded samples:", excluded_distances)

        # 打印实际有多少真实异常被筛除
        count = np.sum(excluded_indices >= 9500)
        print(count)

        return filtered_x, filtered_y

    def extract_features(self, data_loader):
        self.model.eval()
        features = []
        with torch.no_grad():
            for images, _, _ in data_loader:
                images = images.to(self.device)
                feature = self.model.encoder(images)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)

    def save_data(self, filename, data, labels, cluster_targets=None):
        np.savez(filename, data=data.numpy().astype(np.float32), labels=labels.numpy().astype(np.int64), cluster_targets=cluster_targets.numpy().astype(np.int64) if cluster_targets is not None else np.array([]).astype(np.int64))

    def save_datasets(self, train_file, test_file):
        self.save_data(train_file, self.train_set.data, self.train_set.labels, self.train_set.cluster_targets)
        self.save_data(test_file, self.test_set.data, self.test_set.labels)

if __name__ == "__main__":
    root = "./data"
    random_seed = 42
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LeNetAutoencoder().to(device)

    fashion_mnist_dataset = FashionMNIST_Dataset(root, model, device, random_seed)

    train_dataset = fashion_mnist_dataset.train_set
    test_dataset = fashion_mnist_dataset.test_set

    print("train_cluster_target: ", train_dataset.cluster_targets)
    

    train_file = "/data/fmnist/fashionmnist_train_data.npz"
    test_file = "/data/fmnist/fashionmnist_test_data.npz"

    fashion_mnist_dataset.save_datasets(train_file, test_file)