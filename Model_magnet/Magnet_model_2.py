import torch
import torch.nn as nn
import math
import complextorch.nn as compnn
import complextorch
import torch.nn.functional as F
from utils.myBatch import *
import random
import complextorch.nn as cvnn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, use_attention=True):
        super(ChebConv, self).__init__()
        self.use_attention = use_attention

        # Original weights
        self.weight_real = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))
        self.weight_imag = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))

        # Initialize weights
        stdv = 1. / math.sqrt(self.weight_real.size(-1))
        self.weight_real.data.uniform_(-stdv, stdv)
        self.weight_imag.data.uniform_(-stdv, stdv)

        # Normalize magnitude
        magnitude = torch.sqrt(self.weight_real ** 2 + self.weight_imag ** 2)
        self.weight_real.data /= magnitude
        self.weight_imag.data /= magnitude

        # Attention parameters
        if self.use_attention:
            # self.attn_fc = nn.Linear(4 * in_c, 1)  # Concatenates real+imag features
            self.attn_fc = compnn.CVLinear(in_c, 1, bias=bias)
            self.cprelu = compnn.CPReLU()
            self.psoftmax = compnn.PhaseSoftMax(dim=1)

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(1, out_c))
            self.bias_imag = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, data, laplacian):
        X_real, X_imag = data[0], data[1]
        B, N, C = X_real.shape

        # Compute attention weights
        if self.use_attention:
            # Combine real and imaginary features
            # X_combined = torch.cat([X_real, X_imag], dim=-1)  # [B, N, 2C]
            X_combined = X_real + 1j*X_imag

            # Compute attention scores
            X_i = X_combined.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 2C]
            X_j = X_combined.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 2C]
            # concat = torch.cat([X_i, X_j], dim=-1)  # [B, N, N, 4C]
            X = complextorch.CVTensor(X_i, X_j)

            scores = self.attn_fc(X.complex).squeeze(-1)  # [B, N, N]
            scores = self.cprelu(scores)
            attn_weights = self.psoftmax(scores)
            # scores = F.leaky_relu(scores)
            # attn_weights = F.softmax(scores, dim=-1)  # [B, N, N]

            laplacian = laplacian * attn_weights.unsqueeze(1)
            L_real = laplacian.real
            L_imag = laplacian.imag

            # Apply attention to Laplacian
            # L_real = laplacian.real * attn_weights.unsqueeze(1)  # [B, N, N]
            # L_imag = laplacian.imag * attn_weights.unsqueeze(1)  # [B, N, N]
        else:
            L_real = laplacian.real
            L_imag = laplacian.imag

        # Process with attention-adjusted Laplacian
        mul_data = self.process(L_real, L_imag, self.weight_real, self.weight_imag, X_real, X_imag)
        result = torch.sum(mul_data, dim=2)  # Sum over polynomial orders
        # real = result[0] + self.bias_real
        # imag = result[1] + self.bias_imag
        return result[0], result[1]

    def process(self, L_real, L_imag, w_real, w_imag, X_real, X_imag):
        # Batched matrix multiplication
        def bmul(A, B):
            return torch.einsum('bijk,bqjp->bijp', A, B)

        # Real component calculations
        term1_real = bmul(L_real, X_real.unsqueeze(1))
        term1_real = torch.matmul(term1_real, w_real)

        term2_real = -1.0 * bmul(L_imag, X_imag.unsqueeze(1))
        term2_real = torch.matmul(term2_real, w_imag)
        real = term1_real + term2_real

        # Imaginary component calculations
        term1_imag = bmul(L_imag, X_real.unsqueeze(1))
        term1_imag = torch.matmul(term1_imag, w_real)

        term2_imag = bmul(L_real, X_imag.unsqueeze(1))
        term2_imag = torch.matmul(term2_imag, w_imag)
        imag = term1_imag + term2_imag

        return torch.stack([real, imag])


class ChebNet(nn.Module):
    def __init__(self, in_c, num_filter=2, K=2, label_dim=9,
                 dropout=None):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet, self).__init__()

        self.cheb_conv1 = ChebConv(in_c=in_c, out_c=num_filter, K=K, use_attention=False)

        self.cheb_conv2 = ChebConv(in_c=num_filter, out_c=num_filter, K=K, use_attention=False)

        last_dim = 1
        self.dropout = compnn.CVDropout(dropout)
        # for the first loss function on label encoding
        self.linear = nn.Linear(num_filter * 2, num_filter)
        self.conv2 = nn.Conv1d(num_filter * 2, num_filter, kernel_size=1)
        self.conv = compnn.CVConv1d(30 * num_filter, num_filter, kernel_size=1)
        # for the second loss function on class prototypes
        # self.conv = compnn.CVConv1d(30 * last_dim, 128, kernel_size=1)
        self.tanh = compnn.CVPolarTanh()
        self.bn = ComplexBatchNorm1d(30, affine=False)

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, imag, laplacian, layer=2, size=30):

        real, imag = self.cheb_conv1((real, imag), laplacian)
        # print("cheb Conv1:",self.cheb_conv1.weight_real.requires_grad_())
        for l in range(1,layer):
            real, imag = self.cheb_conv2((real, imag), laplacian)
            real, imag = self.complex_relu(real, imag)

        # real, imag = torch.mean(real, dim=2), torch.mean(imag, dim=2)
        x = torch.cat((real, imag), dim=-1)
        x = self.conv2(x.transpose(2, 1)).mean(dim=-1)
        # x = x.reshape(x.shape[0], -1)
        # x = self.linear(x)
        # real, imag = real.reshape(real.shape[0], -1), imag.reshape(imag.shape[0], -1)

        # x = complextorch.CVTensor(real, imag).to(device)
        # x = self.bn(x)
        # x = self.tanh(x)
        # x = self.conv2(x[:, :, None])
        # for the first loss function label encoding
        # return x.squeeze(2)
        # for the second loss function
        return x.squeeze()

class Model(torch.nn.Module):
    def __init__(self, encoder: ChebNet, num_hidden: int, num_proj_hidden: int,  num_label: int, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: ChebNet = encoder
        self.tau: float = tau

        self.cprelu = cvnn.CPReLU()

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        # self.fc1 = cvnn.CVLinear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        # self.fc2 = cvnn.CVLinear(num_proj_hidden, num_hidden)
        self.prediction_layer = torch.nn.Linear(num_proj_hidden*2, num_label)
        # self.prediction_layer = cvnn.CVLinear(num_proj_hidden*2, num_label)
        self.prediction_layer_sdgcn = torch.nn.Linear(num_proj_hidden, num_label)
        # self.prediction_layer_sdgcn = cvnn.CVLinear(num_proj_hidden, num_label)

    def forward(self, real, imag, laplacian, layer=2):
        return self.encoder(real, imag, laplacian, layer=layer)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        # z = self.cprelu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        # return -torch.log(
        #     between_sim.diag()
        #     / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return -torch.log(
            between_sim.diag()
            / (between_sim.sum(1)))

    def batch_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        num_nodes = z1.size(0)
        idx = random.sample(list(range(num_nodes)),batch_size)
        z1, z2 = z1[idx], z2[idx]
        return self.semi_loss(z1,z2)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        # if batch_size == 0:
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

    def label_loss(self, z1, z2, y_label):
        x = torch.cat((z1, z2), dim=-1)
        x = self.prediction_layer(x)
        x = F.log_softmax(x, dim=1)

        loss = F.nll_loss(x, y_label.squeeze().long())
        return loss

    def prediction(self, z1, z2):
        x = torch.cat((z1, z2), dim=-1)
        x = self.prediction_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
