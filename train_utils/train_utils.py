import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.hermitian import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import time
from utils.utils_loss import *
from utils.perturb import *
from Model_magnet.encoding_loss_function import *
from sklearn.metrics import confusion_matrix
from scipy.signal import hilbert
from utils.io_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def contrastive_graph_construction(graphs, K):

    L1, L2 = [], []

    for graph in graphs:

        A = coo_matrix(graph.cpu())
        weights = A.data
        pos_edges = np.stack([A.row, A.col], axis=1)

        pos_edges1 = composite_perturb_unsigned(pos_edges, ratio=0.1)
        pos_edges2 = composite_perturb_unsigned(pos_edges, ratio=0.1)

        q1, q2 = random.sample(np.arange(0, 0.5, 0.1).tolist(), 2)
        q1, q2 = np.pi * q1, np.pi * q2

        her_mat_q1 = np.array(decomp(weights, pos_edges1, q1, norm=True, laplacian=True, max_eigen=2,
                                     gcn_appr=True))

        L1.append(np.array(cheb_poly(her_mat_q1, K)))

        her_mat_q2 = np.array(decomp(weights, pos_edges2, q2, norm=True, laplacian=True, max_eigen=2,
                                     gcn_appr=True))

        L2.append(np.array(cheb_poly(her_mat_q2, K)))

    return np.array(L1), np.array(L2)


def dataloader(data_train, label_train, A_pdc_train, data_val, label_val,  A_pdc_valid, batch_size):

    # feature_imag_train = np.imag(hilbert(data_train, axis=-1))
    feature_imag_train = data_train
    # feature_imag_valid = np.imag(hilbert(data_val, axis=-1))
    feature_imag_valid = data_val

    feature_real_train, feature_imag_train = torch.FloatTensor(data_train.reshape(-1, data_train.shape[-1])).to(device),\
                                             torch.FloatTensor(feature_imag_train.reshape(-1, data_train.shape[-1])).to(device)

    feature_real_valid, feature_imag_valid = torch.FloatTensor(data_val.reshape(-1, data_val.shape[-1])).to(device),\
                                             torch.FloatTensor(feature_imag_valid.reshape(-1, data_val.shape[-1])).to(device)

    dataset_pdc_train = TensorDataset(torch.from_numpy(A_pdc_train).to(device), feature_real_train, feature_imag_train,
                                torch.from_numpy(label_train).to(device))

    dataset_pdc_valid = TensorDataset(torch.from_numpy(A_pdc_valid).to(device), feature_real_valid, feature_imag_valid,
                                torch.from_numpy(label_val).to(device))

    train_loader = DataLoader(dataset_pdc_train, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(dataset_pdc_valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def train_test_split(train_sub, val_sub, feature_de, Adj, label_repeat):

    data_train = feature_de[list(train_sub), :, :]

    data_val = feature_de[list(val_sub), :, :]

    label_train = np.tile(label_repeat, len(train_sub))
    label_val = np.tile(label_repeat, len(val_sub))

    A_pdc_train, A_pdc_valid = Adj[train_sub].reshape([-1, 30, 30]),\
                               Adj[val_sub].reshape([-1, 30, 30])

    return data_train, A_pdc_train, label_train, data_val, A_pdc_valid, label_val


def train_valid(model, optimizer, K, epochs, train_loader, valid_loader, writer=None):

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    epochs_f1, epochs_loss, epochs_metrics, conf_mat_epochs = [], [], [], []

    met_calc = Metrics(num_class=9)

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)

    # Loss = loss_fucntion_2(distance_metric='L2', dist_features=128)

    best_f1, best_err, early_stopping, best_loss = 0, np.inf, 0, 0

    epoch_grads = {}

    for epoch in tqdm(range(epochs)):

        model.train()

        loss_train, train_correct = 0.0, 0.0

        for i, (graphs, X_real, X_imag, label) in enumerate(tqdm(train_loader)):

            label = label.to(device)
            L1, L2 = contrastive_graph_construction(graphs.squeeze(), K)
            L1, L2 = torch.tensor(L1).to(device), torch.tensor(L2).to(device)

            start_time = time.time()
            # graph = torch.rand(*graph.shape, dtype=torch.complex64).to(device)
            # X_real = torch.rand(*X_real.shape).to(device)
            # X_imag = torch.rand(*X_imag.shape).to(device)
            ####################
            # Train
            ####################
            count  = 0.0
            X_real, X_imag = X_real.squeeze().reshape([-1, 30, 5]).to(device), \
                             X_imag.squeeze().reshape([-1, 30, 5]).to(device)
            z1 = model(X_real, X_imag, L1)
            z2 = model(X_real, X_imag, L2)
            # new loss definition
            # train_loss, pred_label= loss_function(criterion, preds, labels, label, beta=beta, test_flag=True)
            # train_loss, pred_label= loss_function(criterion, preds, labels, label, beta=beta, distance_metric='orth')
            contrastive_loss = model.contrastive_loss(z1, z2)
            # contrastive_loss = model.info_nce_loss(z)
            label_loss = model.label_loss(z1, z2, label)
            train_loss = 0.2 * contrastive_loss + label_loss
            loss_train += train_loss.detach().item()
            logits = model.prediction(z1, z2)
            pred_label = torch.argmax(logits, dim=-1)
            train_correct += (pred_label.squeeze() == label).sum().detach().item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()

        loss_valid = 0.0

        pred_, label_ = [], []

        with torch.no_grad():

            for i, (graphs, X_real, X_imag, label) in enumerate(valid_loader):
                label = label.to(device)
                L1, L2 = contrastive_graph_construction(graphs.squeeze(), K)
                L1, L2 = torch.tensor(L1).to(device), torch.tensor(L2).to(device)

                ####################
                # Valid
                ####################
                count = 0.0
                X_real, X_imag = X_real.squeeze().reshape([-1, 30, 5]).to(device), \
                                 X_imag.squeeze().reshape([-1, 30, 5]).to(device)
                z1 = model(X_real, X_imag, L1)
                z2 = model(X_real, X_imag, L2)

                valid_loss = model.label_loss(z1, z2, label)
                logits = model.prediction(z1, z2)
                pred_label = torch.argmax(logits, dim=-1)
                # train_correct += (pred_label.squeeze() == label).sum().detach().item()
                loss_valid += valid_loss.detach().item()
                pred_ += pred_label
                label_ += label.squeeze().tolist()

        final_metrics = met_calc.compute_metrics(torch.tensor([pred_]).to(device),
                                 torch.tensor([label_]).to(device))


        if writer:

            epochs_metrics.append(final_metrics)

            outstrtrain = 'epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                          (epoch, loss_valid / len(valid_loader), final_metrics[0], final_metrics[1], final_metrics[2],
                           final_metrics[3])

            print(outstrtrain)

            pred_np, label_np = np.array(torch.tensor(pred_).tolist()), np.array(torch.tensor(label_).tolist())

            conf_mat_epochs.append(confusion_matrix(label_np, pred_np))

            writer.add_scalars('Loss', {'Train': loss_train / len(train_loader),
                                        'Validation': loss_valid / len(valid_loader)}, epoch)

            writer.add_scalars("Accuracy", {'Train': train_correct / len(train_loader.dataset),
                                            'Valid': final_metrics[0]}, epoch)

            writer.add_scalar("recall/val", final_metrics[1], epoch)
            writer.add_scalar("precision/val", final_metrics[2], epoch)
            writer.add_scalar("F1 Score", final_metrics[3], epoch)

    if writer:

        return epochs_metrics, conf_mat_epochs, epoch_grads

    else:

        return best_f1
