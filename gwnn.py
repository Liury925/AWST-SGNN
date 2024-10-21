import torch
import torch.nn as nn
from torch_sparse import spspmm, spmm
from torch.autograd import Variable
from utils import get_sim
import numpy as np
import tensorflow as tf
from wavelet_filter import *
import torch.nn.functional as F
class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param ncount: Number of nodes.
    :param device: Device to train on.
    """
    def __init__(self, args, elipson,in_channels, out_channels, ncount, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.ncount = ncount
        self.device = device
        self.elipson = elipson
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining scale matrix (Theta in the paper) and weight matrix.
        """
        self.scales = torch.nn.Parameter(torch.Tensor(self.args.num_scales, self.args.num_bands))
        self.low_scale = torch.nn.Parameter(torch.Tensor(self.args.num_scales, 1))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount ,1))
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.scales, 0.1, self.args.max_initial_scale)
        torch.nn.init.uniform_(self.low_scale, 4, 6)
        #torch.nn.init.xavier_uniform_(self.scales)
        #torch.nn.init.xavier_uniform_(self.scales)
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.8, 1.2)


    def compute_filter(self,L,W):
        self.l_band = self.scales[:, :, None] * self.elipson[None, None, :]
        self.l_band = self.l_band.to(self.device)
        self.g_band = mexican_hat_wavelet_tf(self.l_band)
        self.g_band = self.g_band.to(self.device)
        self.l_low = self.low_scale[:, :, None] * self.elipson[None, None, :]
        #self.g_low = low_pass_filter_tf(self.l_low)
        self.g_low = hot_wavelet_tf(self.l_low)
        self.g = torch.cat((self.g_low, self.g_band), dim=1)
        self.g = torch.sum(self.g, dim=1)
        #self.g = torch.sum(self.g_band, dim=1)
        self.g = torch.mean(self.g, dim=0)
        filter = weighted_least_squares(L, W, self.elipson, self.args.k, self.g)
        return filter

    def forward(self, features,L,A,W):
        filter = self.compute_filter(L,W)
        f1 = self.diagonal_weight_filter * filter.t()
        f = torch.mm(filter, f1)
        #f = torch.mm(filter,filter.t())
        #filter = filter.to(self.device)
        f = f.to(self.device)
        P = self.args.gama * f + (1-self.args.gama) * A
        filtered_features = self.args.alpha * torch.mm(P.squeeze(0), features.squeeze(0)) +(1-self.args.alpha)*features.squeeze(0)
        localized_features = torch.mm(filtered_features,self.weight_matrix)
        return localized_features



class GraphWaveletNeuralNetwork(torch.nn.Module):
    """
    Graph Wavelet Neural Network class.
    For details see: Graph Wavelet Neural Network.
    :param args: Arguments object.
    :param ncount: Number of nodes.
    :param feature_number: Number of features.
    :param class_number: Number of classes.
    :param device: Device used for training.
    """
    def __init__(self, elipson, args, ncount, feature_number, device):
        super(GraphWaveletNeuralNetwork, self).__init__()
        self.args = args
        self.ncount = ncount
        self.device = device
        self.layer_num = len(args.cfg)
        self.net = torch.nn.ModuleList()
        in_channel = feature_number
        for i, v in enumerate(args.cfg):
            out_channel = int(v)
            if len(args.cfg) ==1 or i!= (self.layer_num-1):
                self.net.append(GraphWaveletLayer(args, elipson, in_channel, out_channel, self.ncount, self.device))
              #  self.net.append(nn.BatchNorm1d(out_channel, affine = False))
                self.net.append(nn.ReLU())
            else:
                self.net.append(GraphWaveletLayer(args, elipson, in_channel, out_channel, self.ncount, self.device))
            in_channel = out_channel


    def embedding(self, features,L,A,W):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        for i, layer in enumerate(self.net):
            if not isinstance(layer,nn.ReLU):
                features = layer(features,L,A,W)
            else:
                features = layer(features)
                features = torch.nn.functional.dropout(features, training=self.training, p=self.args.dropout)
        return features


    def get_loss(self,z1,z2):
        f = lambda x: torch.exp(x)
        refl_sim = f(get_sim(z1,z1))
        between_sim = f(get_sim(z1,z2))
        #x1 = (refl_sim.sum(1)-refl_sim.diag())/(self.ncount-1) + (between_sim.sum(1) - #between_sim.diag())/(self.ncount-1) + between_sim.diag()
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -torch.log(between_sim.diag() / x1)
        """loss = 0
       # loss1 = 0
        refl_dis = F.pairwise_distance(z1, z2)
        print(refl_dis)
        print("--------------------------------------------")
        lbl_z = torch.tensor([0.]).cuda()
        # margin_label = -1 * torch.ones_like(refl_sim)
        for i in range(self.args.num_n):
            idx_0 = np.random.permutation(len(z2))
            z_n = z2[idx_0].cuda()
            between_dis = F.pairwise_distance(z1, z_n)
            loss += torch.max((refl_dis - between_dis + self.args.margin), lbl_z).mean()
            #loss1 += torch.max((between_dis - refl_dis - self.args.margin1 - self.args.margin2), lbl_z).mean()
            print(between_dis)
        # margin_loss = torch.nn.MarginRankingLoss(margin=self.args.margin1, reduce=False)
        #return loss+loss1"""
        return loss


    def forward(self, features,aug_features,L,A,W):
        """
        Forward propagation pass.
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param feature_indices: Feature matrix index pairs.
        :param feature_values: Feature matrix values.
        :param predictions: Predicted node label vector.
        """
        z1 = self.embedding(features,L,A,W)
        z2 = self.embedding(aug_features,L,A,W)
        l1 = self.get_loss(z1,z2)
        l2 = self.get_loss(z2,z1)
        loss = (l1+l2)*0.5
        return loss.mean()








        
