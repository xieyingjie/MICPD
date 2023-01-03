# -*- coding: utf-8 -*-
import logging
import random
import utils
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
import scipy.io as scio
from itertools import chain
import cStringIO
import TILES as t
import datetime
from datetime import datetime,timedelta
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
import matplotlib.pyplot as plt

def getGraph(filepath):
    Friendfile = open(filepath)
    graph1 = []
    G = nx.Graph()
    for i in Friendfile:
        G.add_edge(int(i.strip().split('\t')[0]),int(i.strip().split('\t')[1]),weight=int(i.strip().split('\t')[2]))
    return G

def getCom(cp):
    cfile = open(cp)
    communities = {}
    for i in cfile:
        communities[int(i.strip().split('\t')[0])] = eval(i.strip().split('\t')[1])
    return communities
def getMer(cp):
    cfile = open(cp)
    merge = {}
    for i in cfile:
        merge[int(i.strip().split('\t')[0])] = eval(i.strip().split('\t')[1])
    return merge
def getSpl(cp):
    cfile = open(cp)
    split = {}
    for i in cfile:
        split[int(i.strip().split('\t')[0])] = eval(i.strip().split('\t')[1])
    return split
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def nodeChanged(community1,community2,split={},merge={}):
    label = {}
    nodes = {}
    for i in community1.keys():
        if community1.has_key(i):
            for node in community1[i]:
                nodes[node] = i
    for i in community2.keys():
        if community2.has_key(i):
            for node in community2[i]:
                if nodes.has_key(node):
                    if nodes[node] < i:
                        label[node] = 1
    return label

def growhappen(community1,community2,split={},merge={}):
    label = {}
    nodes = {}
    for i in community1.keys():
        if community1.has_key(i):
            for node in community1[i]:
                nodes[node] = i
    for i in community2.keys():
        if community2.has_key(i):
            for node in community2[i]:
                if nodes.has_key(node):
                    if nodes[node] < i:
                        label[node] = 1
    return label


def extract_windows(path, window_size, mode="train"):
    #files = os.scandir(path)
    windows = []
    lbl = []
    dataset = sio.loadmat(path+"/usc.mat")

    ts = np.array(dataset['Y'])
    ts = ts[:,0]
    cp = np.array(dataset['L'])
    cp = cp[:,0]

    #cp = pd.read_csv(path+"usc_label.csv")
    num_cp = 0
    #ts = np.sqrt(np.power(x[:, 0], 2) + np.power(x[:, 1], 2) + np.power(x[:, 2], 2))
    for i in range(0, ts.shape[0] - window_size, 5):
        windows.append(np.array(ts[i:i + window_size]))
        # print("TS",ts[i:i+window_size])
        is_cp = np.where(cp[i:i + window_size] == 1)[0]
        if is_cp.size == 0:
            is_cp = [0]
        else:
            num_cp += 1
        lbl.append(is_cp[0])

        # print(is_cp)


    print("number of samples : {} /  number of samples with change point : {}".format(len(windows), num_cp))
    windows = np.array(windows)

    return windows, np.array(lbl)

def load_ds(path, window, mode='train'):

    X, lbl = extract_windows(path, window, mode)

    if mode == "all":
        return X, lbl
    train_size = int(floor(0.8 * X.shape[0]))
    if mode == "train":
        trainx = X[0:train_size]
        trainlbl = lbl[0:train_size]
        idx = np.arange(trainx.shape[0])
        np.random.shuffle(idx)
        trainx = trainx[idx,]
        trainlbl = trainlbl[idx]
        print('train samples : ', train_size)
        return trainx, trainlbl

    else:
        testx = X[train_size:]
        testlbl = lbl[train_size:]
        print('test shape {} and number of change points {} '.format(testx.shape, len(np.where(testlbl > 0)[0])))

        return testx, testlbl


def load_dataset(path, ds_name, win, bs, mode="train"):

    trainx, trainlbl = load_ds(path, window=2 * win, mode=mode)

    trainlbl = trainlbl.reshape((trainlbl.shape[0], 1))
    print(trainx.shape, trainlbl.shape)
    dataset = np.concatenate((trainlbl, trainx), 1)

    print("dataset shape : ", dataset.shape)
    if mode == "test":
        return dataset
    return dataset


def getFeatures(G,G0,community1,label,file):
    features=[]
    for i in community1.keys():
        targetG = G.subgraph(community1[i])
        #community features
        nodeNum = targetG.nodes.__len__()
        edgeNum = targetG.edges.__len__()
        intraEdge = np.true_divide(edgeNum,nodeNum)
        interEdge = np.true_divide(G.edges(targetG.nodes).__len__() - targetG.edges.__len__(),nodeNum)
        activity = np.true_divide(G0.edges(targetG.nodes).__len__(),nodeNum)
        Conduct = nx.conductance(G,targetG.nodes)
        file.write(str([nodeNum,edgeNum,intraEdge,interEdge,activity,Conduct])+'\n')
        features.append([nodeNum,edgeNum,intraEdge,interEdge,activity,Conduct])
    return features

def featureExt(wfile,rfile,a,b):
    file = open(wfile,'w')
    for i in range(a,b):
        p = rfile+"graph-"+str(i)
        G = getGraph(p)
        p0 = rfile+"graph-"+str(i-1)
        G0 = getGraph(p0)
        m = rfile+"merging-"+str(i)
        merge = getMer(m)
        cp = rfile+"strong-communities-"+str(i)
        community1 = getCom(cp)
        cp2 = rfile+"strong-communities-"+str(i+1)
        community2 = getCom(cp2)
        label = nodeChanged(community1,community2,merge,merge)
        feature = getFeatures(G,G0,community1,label,file)
    file.close()


def constructNetwork(tempM):
    graph = nx.from_numpy_matrix(tempM)
    graph = graph.to_directed()
    return graph


def getNetworkFeature(path):
    matrixWhole = scio.mmread(path).todense()
    numSnapshots, numNodesSquare = np.shape(matrixWhole)
    numNodes = int(np.sqrt(numNodesSquare))
    # numNodes = 548
    networkFeature = open("data//networkFeature.txt", "w")
    for i in range(numSnapshots):
        tempmm = np.reshape(matrixWhole[i], (numNodes, numNodes))
        tempmm1 = np.diag(np.diag(tempmm))
        tempmm = tempmm - np.diag(np.diag(tempmm))
        graph = constructNetwork(tempmm)
        EdgeNum = graph.edges.__len__()
        NodeNum = graph.nodes.__len__()
        Beta = np.true_divide(EdgeNum, NodeNum)
        degrees = list(graph.degree())
        tpm = 0
        for de in degrees:
            tpm = de[1] + tpm
        tpm = tpm / int(graph.nodes.__len__())
        AverageClustering = nx.average_clustering(graph)
        Density = np.true_divide(2 * EdgeNum, NodeNum * (NodeNum - 1))  # density
        degreeAssortativityCoefficient = nx.degree_assortativity_coefficient(graph)
        Transitivity = nx.transitivity(graph)
        isConnected = nx.is_connected(graph.to_undirected())
        tran = nx.transitivity(graph)

        print(tran)
        networkFeature.write(
            str(i) + ',' + str(EdgeNum) + ',' + str(NodeNum) + ',' + str(Beta) + ',' + str(tpm) + ',' + str(
                AverageClustering) + ',' + str(
                Density) + ',' + str(degreeAssortativityCoefficient) + ',' + str(Transitivity) + ',' + str(
                isConnected) + ',' + str(tran) + "\n")


def smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, domain, window_size):
    beta = np.quantile(utils.distance(encoded_windows, window_size), 0.95)
    alpha = np.quantile(utils.distance(encoded_windows_fft, window_size), 0.95)
    encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)
    encoded_windows_both = utils.matched_filter(encoded_windows_both, window_size)
    distances = utils.distance(encoded_windows_both, window_size)
    distances = utils.matched_filter(distances, window_size)
    return distances

def change_point_score(distances, window_size):
    prominences = np.array(utils.new_peak_prominences(distances)[0])
    prominences = prominences / np.amax(prominences)
    return np.concatenate((np.zeros((window_size,)), prominences, np.zeros((window_size - 1,))))