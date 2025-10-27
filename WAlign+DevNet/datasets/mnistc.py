import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter

torch.cuda.empty_cache()
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
torch.set_printoptions(threshold=float('Inf'))
np.set_printoptions(threshold=np.inf)

# LeNet Encoder
class LeNetEncoder(nn.Module):
    def __init__(self):
        super(LeNetEncoder, self).__init__()
        
        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with kernel size 2x2
        
        # Adjusted convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 16, 224, 224)
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 32, 224, 224)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 32, 224, 224)
        self.bn2d3 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 64, 224, 224)
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 32, 112, 112)
        self.bn2d5 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 512, bias=False)  # Output dimension: 128
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(512, self.rep_dim, bias=False)  # Output dimension: 64
        self.bn99 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn2d1(self.conv1(x))))  # Output: (batchsize, 16, 112, 112)
        x = self.pool(F.leaky_relu(self.bn2d2(self.conv2(x))))  # Output: (batchsize, 32, 56, 56)
        x = self.pool(F.leaky_relu(self.bn2d3(self.conv3(x))))  # Output: (batchsize, 32, 28, 28)
        x = self.pool(F.leaky_relu(self.bn2d4(self.conv4(x))))  # Output: (batchsize, 64, 14, 14)
        x = self.pool(F.leaky_relu(self.bn2d5(self.conv5(x))))  # Output: (batchsize, 32, 7, 7)

        x = x.view(int(x.size(0)), -1)  # Flatten for fully connected layers
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = torch.sigmoid(self.bn99(self.fc2(x)))  # Output: (batchsize, 64)

        return x
    
# LeNet Decoder
class LeNetDecoder(nn.Module):
    def __init__(self):
        super(LeNetDecoder, self).__init__()
        
        # Layers corresponding to the encoder
        self.fc1 = nn.Linear(128, 512, bias=False)  # 从64个特征恢复到128个特征
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=False)
        
        self.fc2 = nn.Linear(512, 32 * 7 * 7, bias=False)  # 从128个特征恢复到32 * 7 * 7
        self.bn1d2 = nn.BatchNorm1d(32 * 7 * 7, eps=1e-04, affine=False)

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 32, 7, 7)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 64, 14, 14)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 16, 28, 28)
        self.bn2d3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(8, 4, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 16, 56, 56)
        self.bn2d4 = nn.BatchNorm2d(4, eps=1e-04, affine=False)

        self.deconv5 = nn.ConvTranspose2d(4, 3, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 3, 112, 112)
        self.bn2d5 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        # Final output layer to achieve (batchsize, 3, 224, 224)
        self.deconv6 = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=1, padding=2)  # Output: (batchsize, 3, 224, 224)

    def forward(self, x):
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))  # Output: (batchsize, 128)
        x = F.leaky_relu(self.bn1d2(self.fc2(x)))  # Output: (batchsize, 32 * 7 * 7)
        x = x.view(int(x.size(0)), 32, 7, 7)  # Reshape for convolutional layers
        
        x = F.interpolate(F.leaky_relu(self.bn2d1(self.deconv1(x))), scale_factor=2)  # Output: (batchsize, 32, 7, 7)
        x = F.interpolate(F.leaky_relu(self.bn2d2(self.deconv2(x))), scale_factor=2)  # Output: (batchsize, 64, 14, 14)
        x = F.interpolate(F.leaky_relu(self.bn2d3(self.deconv3(x))), scale_factor=2)  # Output: (batchsize, 16, 28, 28)
        x = F.interpolate(F.leaky_relu(self.bn2d4(self.deconv4(x))), scale_factor=2)  # Output: (batchsize, 16, 56, 56)
        x = F.interpolate(F.leaky_relu(self.bn2d5(self.deconv5(x))), scale_factor=2)  # Output: (batchsize, 3, 112, 112)
        x = self.deconv6(x)  # Output: (batchsize, 3, 224, 224)
        
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

class MyMNISTC(Dataset):
    def __init__(self, root, is_train=True, transform=None, target_transform=None, encoder=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        self.semi_targets = []
        self.cluster_targets = []
        self.encoder = encoder
        self._load_data()
        
        if is_train:  # 仅在训练时进行聚类
            self._pretrain_autoencoder()  # 预训练自编码器
            self._perform_clustering()  # 进行聚类

    def _load_data(self):
        if self.is_train:
            labeled_path = os.path.join(self.root, 'labeled')
            unlabeled_path = os.path.join(self.root, 'unlabeled')

            print("Labeled path:", labeled_path)
            print("Unlabeled path:", unlabeled_path)

            # load labeled anomaly
            for img_name in os.listdir(labeled_path):
                img_path = os.path.join(labeled_path, img_name)
                self._append_data(img_path, 1, -1)  # Labeled data with semi_target = -1

            # load unlabeled data
            normal_classes = ['normal_1', 'normal_2', 'normal_3', 'contam']
            # normal_classes = ['normal_1', 'normal_2', 'contam']
            for idx, cls in enumerate(normal_classes, start=1):
                cls_path = os.path.join(unlabeled_path, cls)
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    self._append_data(img_path, 0, idx)
        else:
            test_anomaly_path = os.path.join(self.root, 'anomaly')
            test_normal_path = os.path.join(self.root, 'normal')

            print("Test anomaly path:", test_anomaly_path)
            print("Test normal path:", test_normal_path)

            # Initialize semi_target index
            semi_target_idx = 5

            # Load test seen normal
            seen_normal_path = os.path.join(test_normal_path, 'seen_normal')
            for seen_normal_folder in os.listdir(seen_normal_path):
                seen_normal_folder_path = os.path.join(seen_normal_path, seen_normal_folder)
                for img_name in os.listdir(seen_normal_folder_path):
                    img_path = os.path.join(seen_normal_folder_path, img_name)
                    self._append_data(img_path, 0, semi_target_idx)
                semi_target_idx += 1

            # Load test unseen normal
            unseen_normal_path = os.path.join(test_normal_path, 'unseen_normal')
            for unseen_normal_folder in os.listdir(unseen_normal_path):
                unseen_normal_folder_path = os.path.join(unseen_normal_path, unseen_normal_folder)
                for img_name in os.listdir(unseen_normal_folder_path):
                    img_path = os.path.join(unseen_normal_folder_path, img_name)
                    self._append_data(img_path, 0, semi_target_idx)
                semi_target_idx += 1

            # # Load test seen anomaly
            # seen_anomaly_path = os.path.join(test_anomaly_path, 'seen_anomaly')
            # for seen_anomaly_folder in os.listdir(seen_anomaly_path):
            #     seen_anomaly_folder_path = os.path.join(seen_anomaly_path, seen_anomaly_folder)
            #     for img_name in os.listdir(seen_anomaly_folder_path):
            #         img_path = os.path.join(seen_anomaly_folder_path, img_name)
            #         self._append_data(img_path, 1, semi_target_idx)
            #     semi_target_idx += 1

            # Load test unseen anomaly
            unseen_anomaly_path = os.path.join(test_anomaly_path, 'unseen_anomaly')
            for unseen_anomaly_folder in os.listdir(unseen_anomaly_path):
                unseen_anomaly_folder_path = os.path.join(unseen_anomaly_path, unseen_anomaly_folder)
                for img_name in os.listdir(unseen_anomaly_folder_path):
                    img_path = os.path.join(unseen_anomaly_folder_path, img_name)
                    self._append_data(img_path, 1, semi_target_idx)
                semi_target_idx += 1
            
            self.cluster_targets = [0] * len(self.data)

    def _append_data(self, img_path, label, semi_target):
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            transformed_img = self.transform(img)
        else:
            transformed_img = img
        self.data.append(transformed_img)
        self.labels.append(label)
        self.semi_targets.append(semi_target)

        if not hasattr(self, 'original_data'):
            self.original_data = []
        self.original_data.append(img)

    def _pretrain_autoencoder(self):

        pretrain_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        model = LeNetAutoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        pretrain_data = [pretrain_transform(img) for img in self.original_data]

        train_loader = DataLoader(pretrain_data, batch_size=32, shuffle=True)

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images in train_loader:
                images = images.to(device)

                outputs = model(images)
                loss = criterion(outputs, images)
                loss = torch.mean(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 训练完成后，将编码器设置为提取特征
        self.encoder = model.encoder

    def extract_features(self, data_loader):
        self.encoder.eval()
        features = []
        with torch.no_grad():
            for images in data_loader:
                images = images.to(device)
                feature = self.encoder(images)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)

    def _perform_clustering(self):
        if not self.is_train:
            print("Skipping clustering for test data.")
            self.cluster_targets = [0] * len(self.data)
            return

        # 初始化聚类目标
        cluster_targets = [0] * len(self.data)
        
        # 获取无标签数据的索引
        unlabeled_indices = [i for i, target in enumerate(self.semi_targets) if target != -1]
        
        # 随机生成聚类标签(1或2)
        np.random.seed(42)  # 设置随机种子以保持一致性
        random_clusters = np.random.randint(1, 3, size=len(unlabeled_indices))
        
        # 为无标签数据分配随机聚类标签
        for idx, cluster_label in zip(unlabeled_indices, random_clusters):
            cluster_targets[idx] = cluster_label
        
        # 为标记的异常样本分配-1标签
        labeled_anomaly_indices = [i for i, target in enumerate(self.semi_targets) if target == -1]
        for idx in labeled_anomaly_indices:
            cluster_targets[idx] = -1
            
        self.cluster_targets = cluster_targets
        print("Random clustering completed. Cluster distribution:")
        print(Counter(self.cluster_targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.cluster_targets):
            print(f"Index {idx} out of range for cluster_targets with length {len(self.cluster_targets)}")
        img = self.data[idx]
        label = self.labels[idx]
        semi_target = self.semi_targets[idx]
        cluster_target = self.cluster_targets[idx]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, semi_target, cluster_target

class MNISTC_Dataset:
    def __init__(self, root: str, random_seed: int = 42):
        self.root = root
        self.random_seed = random_seed

        # transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load dataset
        self.train_set = MyMNISTC(root=os.path.join(self.root, 'train'), is_train=True, transform=transform)
        self.test_set = MyMNISTC(root=os.path.join(self.root, 'test'), is_train=False, transform=test_transform)

    def save_data(self, filename, data, labels, semi_targets=None, cluster_targets=None):
        data_tensor = torch.stack(data)
        labels_tensor = torch.tensor(labels)
        semi_targets_tensor = torch.tensor(semi_targets) if semi_targets is not None else torch.tensor([])
        cluster_targets_tensor = torch.tensor(cluster_targets) if cluster_targets is not None else torch.tensor([])
        torch.save({'data': data_tensor, 'labels': labels_tensor, 'semi_targets': semi_targets_tensor, 'cluster_targets': cluster_targets_tensor}, filename)

    def save_datasets(self, train_file, test_file):
        self.save_data(train_file, self.train_set.data, self.train_set.labels, self.train_set.semi_targets, self.train_set.cluster_targets)
        self.save_data(test_file, self.test_set.data, self.test_set.labels, self.test_set.semi_targets, self.test_set.cluster_targets)

    def get_train_set(self):
        return self.train_set
    
    def get_test_set(self):
        return self.test_set

if __name__ == "__main__":
    root = "/home/lgy/myModel/data/mnistc3_con_new"
    random_seed = 42

    mnistc_dataset = MNISTC_Dataset(root, random_seed)

    train_file = "/home/lgy/myModel/random/data/mnistc/unseen/mnistc_train_data.pth"
    test_file = "/home/lgy/myModel/random/data/mnistc/unseen/mnistc_test_data.pth"

    mnistc_dataset.save_datasets(train_file, test_file)

    print("Train set cluster targets:", mnistc_dataset.get_train_set().cluster_targets)
    cluster_targets = mnistc_dataset.get_train_set().cluster_targets
    print(Counter(cluster_targets))
    print("Test set cluster targets:", mnistc_dataset.get_test_set().cluster_targets)
    cluster_targets_test = mnistc_dataset.get_test_set().cluster_targets
    print(Counter(cluster_targets_test))