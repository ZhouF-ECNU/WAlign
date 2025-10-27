import torch
import numpy as np
import os
import pandas as pd
import argparse
from dsad import DeepSAD
from tqdm import tqdm
from metrics import tabular_metrics

torch.cuda.empty_cache()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run_model(epochs, batch_size, lr, result_file):
    random_seed = 42

    clf = DeepSAD(device=device, random_state=random_seed, epochs=epochs, batch_size=batch_size, lr=lr)
    clf.fit(x_train, y=y_train, cluster_labels=train_cluster_labels)

    scores = clf.decision_function(x_test)
    auc, ap, f1 = tabular_metrics(y_test, scores)

    result = {
        'AUC': round(auc, 4),
        'AP': round(ap, 4),
        'F1': round(f1, 4),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr
    }

    print(f"AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    with open(result_file, 'a') as f:
        f.write(f"{result['AUC']},{result['AP']},{result['F1']},{epochs},{batch_size},{lr}\n")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=192, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    args = parser.parse_args()

    # Load training and test data
    train_data = np.load('/home/lgy/myModel/WAlign_code_github/WAlign+DeepSAD/data/train_features.npz')
    x_train = train_data['features']
    y_train = train_data['labels']
    train_cluster_labels = train_data['cluster_labels']

    test_data = np.load('/home/lgy/myModel/WAlign_code_github/WAlign+DeepSAD/data/test_features.npz')
    x_test = test_data['features']
    y_test = test_data['labels']
    test_cluster_labels = test_data['cluster_labels']

    # Result file path
    result_file = '/home/lgy/myModel/WAlign_code_github/WAlign+DeepSAD/results/DeepSAD_results.csv'
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # Write header
    with open(result_file, 'w') as f:
        f.write("AUC,AP,F1,epochs,batch_size,lr\n")

    print(f"Running DeepSAD with epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    run_model(args.epochs, args.batch_size, args.lr, result_file)