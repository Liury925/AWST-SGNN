import json
import pygsp
import numpy as np
import networkx as nx
import torch
from scipy import sparse
from sklearn.preprocessing import normalize
from termcolor import cprint
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import time
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from colorama import Fore
import random




def calculate_mae(predictions, true_values):
    absolute_errors = [abs(pred - true) for pred, true in zip(predictions, true_values)]
    mae = sum(absolute_errors) / len(predictions)
    return mae


def pprint_args(args):
    cprint("Args PPRINT:", 'red', attrs=['bold'])
    for k, v in sorted(args.__dict__.items()):
        print("\t- {}: {}".format(k, v))




def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),),
                         dtype=torch.float32,
                         device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x



def get_sim(z1,z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    sim = torch.mm(z1, z2.t())
    return sim

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def linear_clf(embeddings, y, train_mask, test_mask, class_num, epoch):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = Y[train_mask]
    y_test = Y[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    # with open('{}_embedding.p'.format(dataset), 'ab+') as f:
    #     pkl.dump((y_pred, y_test, degree), f)
    # f.close()

    y_pred = prob_to_one_hot(y_pred)

    acc_list = (np.argmax(y_test, axis=1) == np.argmax(y_pred, axis=1)).astype(float)
    
    acc = np.mean(acc_list) * 100
   
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    
    y_pred1 = torch.where(torch.Tensor(y_pred))
    y_test1 = torch.where(torch.Tensor(y_test))
    y_pred = y_pred1[1].detach().numpy()
    y_test = y_test1[1].detach().numpy()

    mae = calculate_mae(y_pred, y_test)

    class_dict = {}
    for i in range(len(y_test)):
        if y_test[i] not in class_dict:
            class_dict[y_test[i]] = []
        class_dict[y_test[i]].append(acc_list[i])

    for d,l in class_dict.items():
        class_dict[d] = np.mean(l)
        
    bias = np.var(list(class_dict.values()))
    mean = np.mean(list(class_dict.values()))


    string_2 = Fore.GREEN + " epoch: {},accs: {:.2f}, mae:{:.2f} ".format(epoch, acc,mae)
    string_4 = "F1Mi: {},  F1Ma: {}, Mean: {}, Bias: {} ".format(micro, macro, mean, bias)

    return string_2, string_4




