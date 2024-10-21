from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.cm as cmx
import matplotlib.colors as colors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def relation_plot(vectors: np.ndarray, labels: list):
    print(vectors.shape)
    n = len(vectors)
    n_class = len(set(labels))
    corr = np.zeros(shape=(n_class, n_class), dtype=np.float)
    count_matrix = np.zeros(shape=(n_class, n_class))
    distance_matrix = euclidean_distances(vectors, vectors)
    max = distance_matrix.max()
    distance_matrix = distance_matrix/max


    for idx1 in range(n):
        vector1, label1 = vectors[idx1], labels[idx1] - 1
        for idx2 in range(idx1 + 1, n):
            vector2, label2 = vectors[idx2], labels[idx2] - 1
            # 多种metric
            #distance = math.sqrt(np.linalg.norm(vector1 - vector2) / len(vector1))
            #distance = cosine_distances(vector1, vector2)
            distance = distance_matrix[idx1][idx2]
            corr[label1][label2] = (corr[label1][label2] * count_matrix[label1][label2] + distance) / (count_matrix[label1][label2] + 1)
            corr[label2][label1] = corr[label1][label2]
            count_matrix[label1][label2] += 1
            count_matrix[label2][label1] = count_matrix[label1][label2]

    #mask = np.zeros_like(corr)
    #mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(n_class, n_class))
        ax = sns.heatmap(corr, mask=None, vmax=np.max(corr), square=True,
                         cmap="YlGnBu")  #, linewidths=.3 , annot=True, fmt=".3f")
        plt.show()

def plot_networkx(graph, role_labels):
    cmap = plt.get_cmap('tab20')
    x_range = np.linspace(0, 1, len(np.unique(role_labels)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_labels))}
    node_color = [coloring[role_labels[i]] for i in range(len(role_labels))]
    plt.figure()
    nx.draw_networkx(graph, pos=nx.layout.fruchterman_reingold_layout(graph),
                     node_color=node_color, cmap='hot')
    plt.show()
    return

def plot_2D_points(nodes: list, points: np.ndarray, labels: list):
    category_dict = defaultdict(list)
    for idx in range(len(labels)):
        category_dict[int(labels[idx])].append(idx)

    cm = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=len(category_dict))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    plt.figure()
    markers = ['o', '*', 'x', '<', '1', 'p', 'D', '>', '^', 'P', 'X', "v", "s", "+", "d"]

    point_set_list = sorted(category_dict.items(), key=lambda item: item[0])
    for category, point_indices in point_set_list:
        plt.scatter(points[point_indices, 0], points[point_indices, 1],
                    #s=100,
                    marker=markers[category % len(markers)],
                    c=[scalarMap.to_rgba(category)],
                    label=category)

        for label, node_idxs in category_dict.items():
            idx = node_idxs[0]
            x, y = points[idx, 0], points[idx, 1]
            string = ",".join([nodes[idx] for idx in node_idxs])
            plt.text(x, y, s=string)

    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()


# 2D图，karate，没有结构性标签，直接打出node标号
def plot_node_str(nodes: list, points: np.ndarray):
    plt.figure()
    for idx, node in enumerate(nodes):
        x, y = points[idx, 0], points[idx, 1]
        plt.scatter([x], [y])
        plt.text(x, y, s=node)

    plt.xticks([])
    plt.yticks([])
    plt.show()





# PCA降维
def dimension_reduction(features):
    from sklearn.manifold import TSNE
    #model = PCA(n_components=2, whiten=False, random_state=42)
    model = TSNE(n_components=2, perplexity=10, learning_rate=1.0, n_iter=5000)
    results = model.fit_transform(features.detach().cpu().numpy())
    return results




def plot_embeddings(embeddings,y):
    X = []
    Y = []
    for i, v in enumerate(y):
        X.append(i)
        Y.append(v)

    model = TSNE(n_components=2, perplexity=50, n_iter=2000, learning_rate=0.05, random_state=42)
    node_pos = model.fit_transform(embeddings)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()










