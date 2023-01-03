import scipy.io as scio
import networkx as nx
import pandas as pd
import numpy as np
import community as community_louvain
from networkx.algorithms.community import k_clique_communities
import SBM_loader
import community as community_louvain
from sklearn.cluster import SpectralClustering

def main():
    communityfile = open("data//communityGEDInputLOV.txt", "w")
    fname = 'communityAndroleCP_0.05_0.45_0.1.txt'
    edgefile = "Synthetic/" + fname
    G_times = SBM_loader.load_temporarl_edgelist(edgefile)
    i = 0
    communityfile.write('{"windows": [')
    for timeGraph in  G_times:
        # print(i)
        # i = i + 1
        # model = SpectralClustering(n_clusters=10, assign_labels='discretize',
        #                            affinity='precomputed',
        #                            random_state=1).fit(nx.convert_matrix.to_numpy_array(timeGraph))
        #
        # if i == 15:
        #     print("1")
        # c = list(k_clique_communities(timeGraph, 3))
        # outfilegdelist = []
        # print(len(c))
        # for j, list_nodes in enumerate(c):
        #     targetG = timeGraph.subgraph(list_nodes)
        #     edgelist = targetG.edges()
        #     stredge = ''
        #
        #     for edge in edgelist:
        #         toedge = '[' + str(edge[0]) + ',' + str(edge[1]) + ']'
        #         toedge.replace("\n", "")
        #         stredge = stredge + toedge + ","
        #
        #     if j == len(c) - 1:
        #         wedge = '[' + stredge[:-1] + ']'
        #     else:
        #         wedge = '[' + stredge[:-1] + '],'
        #
        #     outfilegdelist.append(wedge)
        #
        # if i == 150:
        #     json_data = '{"communities":[' + ''.join(outfilegdelist) + ']}'
        # else:
        #     json_data = '{"communities":[' + ''.join(outfilegdelist) + ']},'
        # communityfile.write(json_data)
        # i = i+1
    # communityfile.write(']}')
        partition = community_louvain.best_partition(timeGraph)
        j = 0
        outfilegdelist = []
        testjj = 0
        for com in set(partition.values()):
            testjj = testjj + 1
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            # tep = graph.subgraph(list_nodes)
            # # list_nodes1=['194','167','473']
            targetG = timeGraph.subgraph(list_nodes)
            edgelist = targetG.edges()
            stredge = ''

            for edge in edgelist:
                toedge = '[' + str(edge[0]) + ',' + str(edge[1]) + ']'
                toedge.replace("\n", "")
                stredge = stredge + toedge + ","

            if j == len(set(partition.values())) - 1:
                testj = set(partition.values())
                lenj = len(testj)
                wedge = '[' + stredge[:-1] + ']'
            else:
                wedge = '[' + stredge[:-1] + '],'

            j = j + 1
            outfilegdelist.append(wedge)
        # json_data = '{"communities":[' + ''.join(outfilegdelist) + ']},'
        # communityfile.write(json_data)

        print(testjj)
        if i == 150:
            json_data = '{"communities":[' + ''.join(outfilegdelist) + ']}'
        else:
            json_data = '{"communities":[' + ''.join(outfilegdelist) + ']},'
        communityfile.write(json_data)
        i = i+1

    communityfile.write(']}')

if __name__ == "__main__":
    main()