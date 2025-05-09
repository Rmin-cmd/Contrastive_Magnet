import scipy.io as sio
import torch.optim as optim
from utils import load_data
import os
from torch.utils.tensorboard import SummaryWriter
from Model_magnet.Magnet_model_2 import ChebNet, Model
from train_utils.train_utils import *
from scipy.stats import beta


def disparity_filter_adjacency(adj_matrix, alpha=0.05):
    """
    Apply the disparity filter to an adjacency matrix to identify statistically significant edges.

    Parameters:
    - adj_matrix: numpy array, adjacency matrix (weighted, undirected).
    - alpha: significance level (default: 0.05).

    Returns:
    - significant_matrix: numpy array, adjacency matrix with only significant edges retained.
    """
    # Initialize the significant adjacency matrix
    significant_matrix = np.zeros_like(adj_matrix)

    # Iterate over all nodes
    for node in range(adj_matrix.shape[0]):
        # Get the weights of edges connected to the node
        weights = adj_matrix[node, :]
        connected_edges = weights > 0  # Find non-zero edges

        if np.sum(connected_edges) > 1:  # Node needs at least 2 connections for disparity filtering
            # Normalize weights by the total strength of the node
            total_weight = np.sum(weights[connected_edges])
            # normalized_weights = weights[connected_edges] / total_weight

            # Degree of the node (number of edges)
            k = np.sum(connected_edges)

            # Identify significant edges using the null model
            for i, w_norm in zip(np.where(connected_edges)[0], weights/total_weight):
                # Null model: weights follow a uniform distribution split into k subintervals
                p_value = 1 - beta.cdf(w_norm, a=1, b=k - 1)

                # Retain edge if it's significant
                if p_value < alpha:
                    significant_matrix[node, i] = adj_matrix[node, i]

    return significant_matrix


device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_4.mat'

root_dir = r'preprocessed_data\preprocessed_feature\smooth_preprocessed_28' # for 1-second window

log_path = 'saved_model'

n_folds = 10
num_subs = 123
batch_size = 64
lr = 1e-3
l2 = 5e-4
n_per = num_subs // n_folds
categories = 9
epochs = 50
q = 0.01
K = 2

class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']
# label_comp = np.random.permutation(label_comp)

A_pdc = sio.loadmat(data_path)['data']

A_pdc = np.mean(A_pdc, axis=1)

# A_pdc = np.random.random(A_pdc.shape)

# A_pdc = np.zeros(A_pdc.shape)

label_type = 'cls2' if categories == 2 else 'cls9'

criterion = nn.CrossEntropyLoss()

met_calc = Metrics(num_class=9)

acc_fold, recall_fold, precision_fold, f1_score_fold = [], [], [], []


conf_mat_final = []

for fold in tqdm(range(n_folds)):

    # feature_pdc = sio.loadmat(feature_path)['de_fold'+str(fold)]
    data_dir = os.path.join(root_dir,'de_lds_fold%d.mat' % (fold))
    feature_pdc = sio.loadmat(data_dir)['de_lds']
    # feature_pdc = np.random.random(feature_pdc.shape)

    label_repeat = load_data.load_srt_de(feature_pdc, True, label_type, 11)

    if fold < n_folds - 1:
        val_sub = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val_sub = np.arange(n_per * fold, n_per * (fold + 1) - 1)

    train_sub = list(set(np.arange(num_subs)) - set(val_sub))

    data_train, A_pdc_train, label_train, data_test, A_pdc_test, label_test = train_test_split(
        train_sub,
        val_sub, feature_pdc,
        A_pdc, label_repeat)

    train_loader, valid_loader = dataloader(data_train, label_train, A_pdc_train, data_test, label_test, A_pdc_test, batch_size=batch_size)

    encoder = ChebNet(5, num_filter=64, dropout=0.2, K=K).to(device)

    model = Model(encoder, num_hidden=64, num_proj_hidden=64, num_label=9)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=f"runs/FCMagnet_zero_2/fold_{fold}")

    met_epochs, conf_mat_epochs, epoch_grads = train_valid(model, optimizer, K, epochs=50, train_loader=train_loader,
                                              valid_loader=valid_loader, writer=writer)

    # for key in epoch_grads.keys():
    #     df = pd.DataFrame(epoch_grads[key])
    #     df.boxplot()
    #     plt.title(key)
    #     plt.xlabel('epochs')
    #     plt.ylabel('gradients')
    #     plt.show()

    fig = show_confusion(np.mean(conf_mat_epochs, axis=0), class_names, show=False)

    writer.add_figure("Confusion Matrix", fig)

    acc_fold.append(np.max(met_epochs, axis=0)[0]), recall_fold.append(np.max(met_epochs, axis=0)[1]),
    precision_fold.append(np.max(met_epochs, axis=0)[2]), f1_score_fold.append(np.max(met_epochs, axis=0)[3])

    outstrtrain = 'fold:%d, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                  (fold, acc_fold[fold], recall_fold[fold], precision_fold[fold], f1_score_fold[fold])

    print(outstrtrain)


print('folds accuracy: %.3f ± %.3f, folds recall: %.3f ± %.3f, folds precision: %.3f ± %.3f, folds F1-Score: %.3f ± %.3f' %
      (np.mean(acc_fold), np.std(acc_fold), np.mean(recall_fold), np.std(recall_fold), np.mean(precision_fold),
       np.std(precision_fold), np.mean(f1_score_fold), np.std(f1_score_fold)))


