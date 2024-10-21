import numpy as np
import torch
import networkx as nx
from scipy.special import chebyt
from scipy.interpolate import PchipInterpolator

"""------------------------------------------------------------------------------"""
def get_norm_laplacian_matrix(nx_graph):
    L = nx.laplacian_matrix(nx_graph)
    degrees = np.array(nx_graph.degree())[:, 1]
    degrees_nonzero = degrees.copy()
    degrees_nonzero[degrees_nonzero == 0] = 1  # 将度数为零的节点设为1
    D_sqrt = np.diag(np.power(degrees_nonzero, -0.5))
    L_hat = D_sqrt.dot(L.toarray()).dot(D_sqrt)
    L_hat = torch.Tensor(L_hat)
    return L_hat

def get_norm_adj_matrix(nx_graph):
    A = nx.adjacency_matrix(nx_graph)
    degrees = np.array(nx_graph.degree())[:, 1]
    degrees_nonzero = degrees.copy()
    degrees_nonzero[degrees_nonzero == 0] = 1  # 将度数为零的节点设为1
    D_sqrt = np.diag(np.power(degrees_nonzero, -0.5))
    A_hat = D_sqrt.dot(A.toarray()).dot(D_sqrt)
    A_hat = torch.Tensor(A_hat)
    return A_hat

def remove_zeros(tensor):
    non_zeros = tensor[tensor != 0]
    return non_zeros

"""wavelet_utils"""
"""------------------------------------------------------------------------------"""
def mexican_hat_wavelet_tf(l):
    """
    Generating Mexican Hat Wavelets.
    l: Product of scale and eigenvalues.
    """
    const = torch.tensor(2.0 * np.sqrt(2.0 / 3.0) * np.power(np.pi, -0.25))
    val = const * (1 - l ** 2) * torch.exp(-0.5 * l ** 2)
    return val

def hot_wavelet_tf(l):
    return torch.exp(-1 * l)


def low_pass_filter_tf(l):
    """
    Generating a proportional function as a low-pass filter.
    l: Product of scale and eigenvalues.
    """
    return 1.0 / (1.0 + l)




"""approximation_utils"""
"""------------------------------------------------------------------------------"""
# Create a polynomial function
def poly1d(x, coefficients):
    result = torch.zeros_like(x)
    power = x.clone().detach()  # 复制x并断开与计算图的连接
    for coeff in coefficients:
        result = result + coeff * power
        power = power * x
    return result



def bandpass_chebychev_coefficients(L,bandpass_range, m):
    """
    Approximate an ideal bandpass filter using Chebychev or Jackson-Chebychev
    polynomials.

    Inputs:
        - `lmax`: estimation of the largest eigenvalue of the Laplacian.
        - `bandpass_range = (fmin, fmax)`: bandpass frequency. We should have \
        fmin >= 0 and fmax <= lmax.
        - `m`: order of the polynomial approximation

    Outputs:
        - `CH`: array of m+1 Chebychev coefficients approximating the desired \
        filter.
        - `JCH`: array of m+1 Jackson-Chebychev coefficients approximating the \
        desired filter.

    For more details about these polynomial approximations please refer to:
        - Napoli et al., `Efficient estimation of eigenvalue counts in an \
        interval`, Numerical Linear Algebra with Applications, 23(4), 674–692, \
        2016.
    """

    # Check validity of bandpass range
    lmax = 2.0
    lmin = 0.0
    assert bandpass_range <= lmax, 'Bandpass range should be in [0, lmax]'

    # Scaling and translation to come back to the classical interval
    # of Chebychev polynomials [-1,1]
    mid = lmax / 2
    a = (lmin - mid) / mid
    b = (bandpass_range - mid) / mid

    # Compute Chebychev coefficients that approximate the bandpass filter
    CH = np.zeros(m + 1)
    CH[0] = (1. / np.pi) * (np.arccos(a) - np.arccos(b))
    for j in range(1, m + 1):
        CH[j] = (2. / (np.pi * j)) * (np.sin(j * np.arccos(a)) - np.sin(j * np.arccos(b)))

    # Compute Jackson Chebychev coefficients that approximate the bandpass filter
    gamma = np.zeros(m + 1)
    alpha = np.pi / (m + 2)
    for j in range(0, m + 1):
        gamma[j] = (1. / np.sin(alpha)) * (
                (1. - float(j) / (m + 2)) * np.sin(alpha) * np.cos(j * alpha) + \
                (1. / (m + 2)) * np.cos(alpha) * np.sin(j * alpha))
    JCH = np.multiply(CH, gamma);
    JCH = torch.Tensor(JCH)
    theta = poly1d(L, JCH)
    return theta



def generate_rademacher_vector(size):
    """
    Generate identically independently distributed Rademacher random variables, where each entry of a randomly generated vector assumes values -1 and 1 with equal probability 1/2.
    Need to generate R times, taking the mean.
    """
    result = np.random.choice([-1, 1], size=(size,1), p=[0.5, 0.5])
    result = torch.Tensor(result)
    return result


def interpolating(x, y, num):
    """
    An approximation of the cumulative spectral density function using monotonic interpolation.
    """
    interpolator = PchipInterpolator(x, y.cpu())
    xi = np.linspace(min(x), max(x), num)
    yi = interpolator(xi)
    return xi, yi


#
def spectral_density_function(L,R,N,elipson,k):
    """
    Calculate an approximation of the spectral density.
    """
    P = []
    for i in range(len(elipson)):
        theta = bandpass_chebychev_coefficients(L.cpu(), elipson[i].cpu(),2)
        theta = theta.to('cpu')
        sum = 0.0
        for r in range(R):
            z = generate_rademacher_vector(theta.shape[1])
            z = z.to('cpu')
            sum = sum + torch.mm(z.t(), torch.mm(theta, z))
        sum = sum / R
        P.append(sum / N)
    P = torch.Tensor(P)
    xi, pi = interpolating(elipson.cpu(), P, 1000)
    coefficients = np.polyfit(xi, pi, k)
    derivative_coefficients = np.polyder(coefficients)
    w = poly1d(elipson, derivative_coefficients)
    w = torch.diag(w)
    return w


def weighted_least_squares(L,W,elipson,k,g):
    """
    Using the least squares approximation.
    """
    X = torch.vander(elipson, k + 1, increasing=True)
    # Matrix multiplications
    X_tilde = torch.matmul(W, X)
    y_tilde = torch.matmul(W, g)
    # Solve for beta
    beta = torch.matmul(torch.inverse(torch.matmul(X_tilde.T, X_tilde)), torch.matmul(X_tilde.T, y_tilde))
    # Create polynomial
    filter = poly1d(L,beta.cpu())
    return filter

def compute_appr_filter(e,W, scales,low_scales,elipson):
    l_band = elipson[None, None, :] / scales[:, :, None]
    g_band = mexican_hat_wavelet_tf(l_band)
    l_low = low_scales[:, :, None] * elipson[None, None, :]
    g_low = hot_wavelet_tf(l_low)
    g = torch.cat((g_low, g_band), dim=1)
    g = torch.sum(g, dim=1)
    g = torch.mean(g, dim=0)
    f = weighted_least_squares(e, W, elipson, 3, g)

    g_low = torch.mean(g_low[0], dim=0)
    f_low = weighted_least_squares(e, W, elipson, 3, g_low)

    g_band1 = torch.mean(g_band[0], dim=0)
    g_band2 = torch.mean(g_band[1], dim=0)
    f_band1 = weighted_least_squares(e, W, elipson, 3, g_band1)
    f_band2 = weighted_least_squares(e, W, elipson, 3, g_band2)

    return f, f_low, f_band1, f_band2

def compute_true_filter(scales,low_scales,e,u):
    l_band =  e[None, None, :] / scales[:, :, None]
    g_band = mexican_hat_wavelet_tf(l_band)
    l_low = low_scales[:, :, None] * e[None, None, :]
    g_low = hot_wavelet_tf(l_low)
    g = torch.cat((g_low, g_band), dim=1)
    g = torch.sum(g, dim=1)
    g = torch.mean(g, dim=0)


    g_low = torch.mean(g_low[0], dim=0)

    g_band1 = torch.mean(g_band[0], dim=0)
    g_band2 = torch.mean(g_band[1], dim=0)

    """f = torch.mm(u,torch.mm(g,u.t()))
    f_low = torch.mm(u,torch.mm(g_low,u.t()))
    f_band1 = torch.mm(u,torch.mm(g_band1,u.t()))
    f_band2 = torch.mm(u, torch.mm(g_band2, u.t()))"""
    return g, g_low, g_band1, g_band2








