import torch
from torch_geometric.utils import is_undirected, to_undirected  #处理数据集
import os.path as osp   #路径
import numpy as np  #处理数组
from scipy import sparse
from torch_geometric.data import Data  #处理成特定的程序类型
from torch_geometric.datasets import Planetoid, Amazon,Coauthor, KarateClub,WikiCS
#from scipy.sparse import coo_matrix, csr_matrix  #把稀疏的np.array压缩，用法：https://blog.csdn.net/weixin_41894030/article/details/104086803
from torch_geometric.utils import to_networkx
#from data_unit.utils import blind_other_gpus, row_normalize, sparse_mx_to_torch_sparse_tensor,graph_normalize
import os
import scipy.io
import networkx as nx
from torch.utils.data import Dataset



def get_dataset(dataset_name):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'CS', 'Physics','Computers','karate','WikiCS']:
        path = osp.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data/')
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(path, dataset_name)
        elif dataset_name in ['Photo','Computers']:
            dataset = Amazon(path, dataset_name)  # transform=T.ToSparseTensor(),
        elif dataset_name in ['CS','Physics']:
            dataset = Coauthor(path, dataset_name)
        elif dataset_name in ['karate']:
            dataset = KarateClub()
        elif dataset_name == 'WikiCS':
            dataset = WikiCS(root=path + '/' + dataset_name)
        data = dataset[0]
        nx_graph = to_networkx(data)
        nx_graph = nx_graph.to_undirected()

        
        print("Graph info of {}".format(dataset_name))
        print(f'\t- Number of node: {data.num_nodes}')  # 同样是节点属性的维度
        print(f'\t- Number of node features: {data.num_features}')  # 同样是节点属性的维度
        print(f'\t- Number of edge: {data.num_edges}')
        print(f'\t- Number of edge features: {data.num_edge_features}')  # 边属性的维度
        print(f'\t- Average node degree: {data.num_edges / data.num_nodes:.2f}')  # 平均节点度
        print(f'\t- Contains isolated nodes: {data.has_isolated_nodes()}')  # 此图是否包含孤立的节点
        print(f'\t- Contains self-loops: {data.has_self_loops()}')  # 此图是否包含自环的边
        print(f'\t- Is undirected: {data.is_undirected()}')  # 此图是否是无向图
        return data.num_nodes, data.num_features, nx_graph,  data.x, data.y

    elif dataset_name in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:  
        fulldata = scipy.io.loadmat(f'/home/liuruyue/code/IJCAI2023_code/dataset/{dataset_name}.mat')
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = np.array(fulldata['label'], dtype=np.int).flatten()
        num_nodes = node_feat.shape[0]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_feat = torch.tensor(node_feat, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        v = torch.FloatTensor(torch.ones([edge_index.shape[1]]))
        i = torch.LongTensor(np.array([edge_index[0].numpy(), edge_index[1].numpy()]))                   
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([num_nodes, num_nodes]))
        A = A_sp.to_dense()

        
        print("Graph info of {}".format(dataset_name))
        print(f'\t- Number of node: {num_nodes}')  # 同样是节点属性的维度
        print(f'\t- Number of node features: {node_feat.shape[1]}')  # 同样是节点属性的维度
        print(f'\t- Number of edge: {edge_index.shape[1]}')
       
        nx_graph = nx.from_numpy_array(A.cpu().numpy())
        nx_graph = nx_graph.to_undirected()
    
        return num_nodes, node_feat.shape[1], nx_graph, node_feat, label
    

    else:
        raise NotImplementedError


class GraphDataset(Dataset):
    def __init__(self, laplacian_matrix, feature_matrix, adjacency_matrix):
        self.laplacian_matrix = laplacian_matrix
        self.feature_matrix = feature_matrix
        self.adjacency_matrix = adjacency_matrix

    def __getitem__(self, index):
        return self.laplacian_matrix, self.feature_matrix, self.adjacency_matrix

    def __len__(self):
        return 1
