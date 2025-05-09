import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix
import torch
###########################################
####### Dense implementation ##############
###########################################
def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian[0] += np.eye(N, dtype=np.float32)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] += A
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] += 2 * np.dot(A, multi_order_laplacian[k-1]) - multi_order_laplacian[k-2]

    return multi_order_laplacian


# def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
#     A = 1.0*np.array(A)
#     if gcn_appr:
#         A += 1.0*np.eye(A.shape[0])
#
#     A_sym = 0.5*(A + A.T) # symmetrized adjacency
#     # A_sym = 0.7 * A + 0.3 * A.T
#
#     if norm:
#         d = np.sum(np.array(A_sym), axis = 0)
#         d[d == 0] = 1
#         d = np.power(d, -0.5)
#         D = np.diag(d)
#         A_sym = np.dot(np.dot(D, A_sym), D)
#
#     if laplacian:
#         # Theta = 2*np.pi*q*1j*(A - A.T) # phase angle array
#         A_coord = coo_matrix(A)
#         phase_pos = coo_matrix((np.ones(len(A_sym.shape[0])), (A_sym.shape[0], A_sym.shape[0])), shape=(size, size), dtype=np.float32)
#         theta_pos = q * 1j * phase_pos
#         theta_pos.data = np.exp(theta_pos.data)
#         theta_pos_t = -q * 1j * phase_pos.T
#         theta_pos_t.data = np.exp(theta_pos_t.data)
#
#         data = np.concatenate((theta_pos.data, theta_pos_t.data))
#         theta_row = np.concatenate((theta_pos.row, theta_pos_t.row))
#         theta_col = np.concatenate((theta_pos.col, theta_pos_t.col))
#
#         phase = coo_matrix((data, (theta_row, theta_col)), shape=(size, size), dtype=np.complex64)
#         Theta = phase
#         if norm:
#             D = np.diag([1.0]*len(d))
#         else:
#             d = np.sum(np.array(A_sym), axis = 0) # diag of degree array
#             D = np.diag(d)
#         L = D - np.exp(Theta)*A_sym
#     '''
#     else:
#         #transition matrix
#         d_out = np.sum(np.array(A), axis = 1)
#         d_out[d_out==0] = -1
#         d_out = 1.0/d_out
#         d_out[d_out<0] = 0
#         D = np.diag(d_out)
#         L = np.eye(len(d_out)) - np.dot(D, A)
#     '''
#     if norm:
#
#         L = (2.0/max_eigen)*L - np.diag([1.0]*len(A))
#
#     return L

def decomp(weights, edges, q, norm, laplacian, max_eigen, gcn_appr, size=30):

    pos_row, pos_col = edges[:, 0], edges[:, 1]


    A = coo_matrix((weights,
                    (
                        pos_row,
                        pos_col)
                    ),
                   shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:

        phase_pos = coo_matrix((np.ones(len(pos_row)), (pos_row, pos_col)), shape=(size, size), dtype=np.float32)
        theta_pos = q * 1j * phase_pos
        theta_pos.data = np.exp(theta_pos.data)
        theta_pos_t = -q * 1j * phase_pos.T
        theta_pos_t.data = np.exp(theta_pos_t.data)

        data = np.concatenate((theta_pos.data, theta_pos_t.data))
        theta_row = np.concatenate((theta_pos.row, theta_pos_t.row))
        theta_col = np.concatenate((theta_pos.col, theta_pos_t.col))

        phase = coo_matrix((data, (theta_row, theta_col)), shape=(size, size), dtype=np.complex64)
        Theta = phase

        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym)  # element-wise

    if norm:
        L = (2.0 / max_eigen) * L - diag
    return L.toarray()


def hermitian_decomp(As, q = 0.25, norm = False, laplacian = True, max_eigen = None, gcn_appr = False):
    ls, ws, vs = [], [], []
    if len(As.shape)>2:
        for i, A in enumerate(As):
            l, w, v = decomp(A, q, norm, laplacian, max_eigen, gcn_appr)
            vs.append(v)
            ws.append(w)
            ls.append(l)
    else:
        ls, ws, vs = decomp(As, q, norm, laplacian, max_eigen, gcn_appr)
    return np.array(ls), np.array(ws), np.array(vs)
