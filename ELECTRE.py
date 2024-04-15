import pandas as pd
import numpy as np


class ELECTRE:
    def __init__(self):
        self.data = None
        self.c = None
        self.D = None
        self.C = None
        self.sigma = None

    
    def load_data(self, filename):
        data = pd.read_csv(filename, sep=',')
        data.pop('Lp.')
        self.data = data

    
    def gain_type_concordance(self, a, b, p, q):
        # Współczynnik zgodności dla kryterium typu zysk
        if a - b >= -q:
            return 1
        elif a - b < -p:
            return 0
        else:
            return (p - (b - a))  / (p - q)
        
    
    def cost_type_concordance(self, a, b, p, q):
        # Współczynnik zgodności dla kryterium typu koszt
        if a - b <= q:
            return 1
        elif a - b > p:
            return 0
        else:
            return (p - (a - b)) / (p - q)
        

    def gain_type_discordance(self, a, b, p, v):
        # Współczynnik niezgodności dla kryterium typu zysk
        if a - b <= -v:
            return 1
        elif a - b >= -p:
            return 0
        else:
            return ((b - a) - p)  / (v - p)
    

    def cost_type_discordance(self, a, b, p, v):
        # Współczynnik niezgodności dla kryterium typu koszt
        if a - b >= v:
            return 1
        elif a - b <= p:
            return 0
        else:
            return (v - (a - b)) / (v - p)
        

    def count_marginal_concordace(self, p, q):
        struct = np.zeros((self.data.shape[0], self.data.shape[0], self.data.shape[1]))

        for i, a in enumerate(self.data.values):
            for j, b in enumerate(self.data.values):
                if i == j:
                    struct[i, j, :] = 1
                    continue
                for k, g in enumerate(a):
                    if k != 1:
                        struct[i, j, k] = self.gain_type_concordance(g, b[k], p[k], q[k])
                    else:
                        struct[i, j, k] = self.cost_type_concordance(g, b[k], p[k], q[k])

        self.c = struct


    def count_marginal_discordace(self, p, v):
        struct = np.zeros((self.data.shape[0], self.data.shape[0], self.data.shape[1]))

        for i, a in enumerate(self.data.values):
            for j, b in enumerate(self.data.values):
                if i == j:
                    struct[i, j, :] = 0
                    continue
                for k, g in enumerate(a):
                    if k != 1:
                        struct[i, j, k] = self.gain_type_discordance(g, b[k], p[k], v[k]) if self.c[i, j, k] == 0 else 0
                    else:
                        struct[i, j, k] = self.cost_type_discordance(g, b[k], p[k], v[k]) if self.c[i, j, k] == 0 else 0

        self.D = struct


    def count_total_concordance(self, w):
        self.C =  np.sum(self.c * w, axis=2) / np.sum(w)
        

    def count_outranking_credibility(self):
        struct = np.zeros_like(self.C)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                f = np.where(self.d[i, j, :] > self.C[i, j])
                if len(f[0]) == 0:
                    struct[i, j] = self.C[i, j]
                else:
                    if np.any(self.d[i, j, f] == 1):
                        struct[i, j] = 0
                    else:
                        struct[i, j] = self.C[i, j] * np.prod((1 - self.d[i, j, f]) / (1 - self.C[i, j]))

        self.sigma = struct