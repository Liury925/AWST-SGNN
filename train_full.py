import numpy as np
import tensorflow as tf
import torch

from gwnn import *
from param_parser import parameter_args
from dataset_unit import *
from time import perf_counter as t
from tqdm import tqdm
from utils import *
from cluster import *
from plot import *
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    Graph_name = 'CiteSeer'
    args = parameter_args()
    pprint_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cprint("## Loading Dataset ##", "yellow")
    ncount, feature_number, nx_graph, features, target = get_dataset(Graph_name)
    # nx_graph = random_remove_edges(nx_graph, 0.8)
    features =features.to(device)
    target = torch.LongTensor(target).to(device)
    class_number = int(target.max() - target.min()) + 1
    L = get_norm_laplacian_matrix(nx_graph)
    L = L.to(device)
    A = get_norm_adj_matrix(nx_graph)
    A = A.to(device)


    elipson = np.linspace(0, 2, args.S)
    elipson = torch.Tensor(elipson).to(device)
    W = spectral_density_function(L, args.R, ncount, elipson, args.k)
    W = W.to(device)

    random_split = np.random.permutation(ncount)
    train_index = random_split[:int(ncount * args.train_ratio)]
    test_index = random_split[int(ncount * args.train_ratio):]


    model = GraphWaveletNeuralNetwork(elipson,args, ncount, feature_number, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    losses = []
    weight_updates = [[] for _ in range(len(args.cfg))]

    print("Training.\n")
    start = t()

    for current_iter, epoch in enumerate(tqdm(range(0, args.epochs + 1))):
        model.train()
        aug_feature1 = drop_feature(features, args.feature_mask)
        aug_feature1 = aug_feature1.to(device)

        aug_feature2 = drop_feature(features, args.feature_mask)
        aug_feature2 = aug_feature2.to(device)

        loss = model(aug_feature1, aug_feature2,L,A,W)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        string_1 = " ||loss: {:.3f}||".format(loss.item())

        if epoch % args.epochs == 0 and epoch != 0:
            print("\n==========Training End==========\n")
            noe = t()
            print('total time', noe - start)
            print("\n==========Scoring==========\n")

            model.eval()
            embedding = model.embedding(features,L,A,W)
            embs = embedding / embedding.norm(dim=1)[:, None]
            string_2, string_4 = linear_clf(embedding, target, train_index, test_index, class_number, epoch)
            tqdm.write(string_1 + string_2 + string_4+"\n")
            kmeans = KMeans(n_clusters=class_number)
            kmeans.fit(embedding.cpu().detach().numpy())
            y_pred = kmeans.predict(embedding.cpu().detach().numpy())
            cm = clustering_metrics(target.cpu().numpy(), y_pred)
            acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()


    """nodes1 = [str(i) for i in range(ncount)]
    vectors = dimension_reduction(embs)
    relation_plot(embs.detach().cpu().numpy(), target.tolist())
    plot_embeddings(embs[test_index].detach().cpu().numpy(), target[test_index].tolist())
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()"""



