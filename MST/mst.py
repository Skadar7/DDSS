import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


class Graph():
    INF = 999999

    def __init__(self, num_vertices):
        self.V = num_vertices
        self.graph = [[0 for column in range(num_vertices)] for row in range(num_vertices)]

    def printMST(self, parent):
        print("Edge     Weight")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i}       {self.graph[i][parent[i]]}")

    def minKey(self, key, mstSet):
        min = self.INF
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def prims(self):
        key = [self.INF for _ in range(self.V)]
        parent = [None for _ in range(self.V)]
        key[0] = 0
        mstSet = [False for _ in range(self.V)]
        parent[0] = -1

        for _ in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True

            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.printMST(parent)
        return parent



if __name__ == '__main__':
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                             random_state=0)
    #
    # cut_off_dist = 1.
    #
    # gr = pairwise_distances(X, metric='euclidean')

    g = Graph(5)

    g.graph = [[0, 2, 0, 6, 0],
               [2, 0, 3, 8, 5],
               [0, 3, 0, 0, 7],
               [6, 8, 0, 0, 9],
               [0, 5, 7, 9, 0]]

    p = g.prims()
    # p_n = np.array(p)[1:]
    # mst = g.graph[p_n]
    #
    # print(mst[0][0])
