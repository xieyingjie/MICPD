# -*- coding: UTF-8 -*-
'''
dataset generator for synthetic SBM dataset
'''

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io as io
import numpy as np
import pylab as plt
import random
import dateutil.parser as dparser
from networkx.utils import *
import re
import networkx as nx
from networkx import generators
import copy
from scipy.io import savemat

'''
t u v w
'''


def to_edgelist(G_times, outfile):
    outfile = open(outfile, "w")
    tdx = 0
    for G in G_times:

        for (u, v) in G.edges:
            outfile.write(str(tdx) + "," + str(u) + "," + str(v) + "\n")
        tdx = tdx + 1
    outfile.close()
    print("write successful")


def to_edgelist1(G_times, outfile):
    outfile = open(outfile, "w")
    tdx = 0
    list = []
    for G in G_times:
        tempcun = nx.to_numpy_matrix(G)
        list.append(tempcun)
    file_name = 'data.mat'
    savemat(file_name, {'a': list})
    outfile.close()
    print("write successful")


'''
generate ER graph snapshot 
https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.generators.random_graphs.gnp_random_graph.html#networkx.generators.random_graphs.gnp_random_graph
'''

def ER_snapshot(G_prev, alpha, p):
    '''
    for all pairs of nodes, keep its status from time t-1 with 1-alpha prob and resample with alpha prob
    '''
    G_t = G_prev.copy()
    G_new = generators.gnp_random_graph(500, p, directed=False)
    n = 500
    for i in range(0, n):
        for j in range(i + 1, n):
            # remain the same if prob > alpha
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i, j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i, j)
                if (not G_new.has_edge(i, j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i, j)
    return G_t


def SBM_snapshot(G_prev, alpha, sizes, probs):
    G_t = G_prev.copy()
    nodelist = list(range(0, sum(sizes)))
    G_new = nx.stochastic_block_model(sizes, probs, nodelist=nodelist)
    n = len(G_t)
    if (alpha == 1.0):
        return G_new

    for i in range(0, n):
        for j in range(i + 1, n):
            # randomly decide if remain the same or resample
            # remain the same if prob > alpha
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i, j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i, j)
                if (not G_new.has_edge(i, j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i, j)
    return G_t

'''
blocks is an array of sizes
inter is the inter community probability
intra is the intra community probability
'''

def construct_SBM_block(blocks, inter, intra):
    probs = []
    for i in range(len(blocks)):
        prob = [inter] * len(blocks)
        prob[i] = intra
        probs.append(prob)
    return probs

'''
generate both change points based on community and node

alpha = percent of connection resampled, alpha=0.0 means all edges are carried over
'''

def Change_Based_communityAndNode(inter_prob, intra_prob, alpha, incrementintra, incrementintre):
    cps = [16,  34, 61, 87 ]
    fname = "community0.49" + str(inter_prob) + "_" + str(intra_prob) + "_" + str(alpha) + ".txt"

    sizes_2 = [125, 125, 125, 125]
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

    sizes_3 = [50] * 10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    sizes_4 = [62,62,62,62,63,63,63,63 ]
    probs_4 = construct_SBM_block(sizes_4, inter_prob, intra_prob)

    sizes_5 = [83,83,83,83,84,84]
    probs_5 = construct_SBM_block(sizes_5, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)
    list_sizes.append(sizes_4)
    list_sizes.append(sizes_5)

    list_probs = []
    list_probs.append(probs_2)
    list_probs.append(probs_3)
    list_probs.append(probs_4)
    list_probs.append(probs_5)

    list_idx = 0
    sizes = sizes_2
    probs = probs_2

    maxt = 100
    G_0 = nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):
            if t == 16:
                print('community change')
                if ((list_idx + 1) > len(list_sizes) - 1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t,  0.49, sizes, probs)
                G_times.append(G_t)


            elif t == 34:
                print('community change')
                if ((list_idx + 1) > len(list_sizes) - 1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t,  0.49, sizes, probs)
                G_times.append(G_t)

            elif t == 61:
                print('community change')
                if ((list_idx + 1) > len(list_sizes) - 1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, 0.49, sizes, probs)
                G_times.append(G_t)

            else:
                print('community change')
                if ((list_idx + 1) > len(list_sizes) - 1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t,  0.49, sizes, probs)
                G_times.append(G_t)

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            G_times.append(G_t)
    to_edgelist(G_times, fname)
    ii = 1
    for G in G_times:
        aj = np.array(nx.adjacency_matrix(G).todense())
        aj1 = np.array(nx.adjacency_matrix(G_times[ii]).todense())
        r = cosine_similarity(aj, aj1)
        s = np.sum(r)
        print(s)
        ii = ii + 1


def main():

    inter_prob = 0.05
    intra_prob = 0.45
    incrementintra = 0.2
    incrementintre = 0.2
    alpha = 0.1
    Change_Based_communityAndNode(inter_prob, intra_prob, alpha, incrementintra, incrementintre)


if __name__ == "__main__":
    main()
