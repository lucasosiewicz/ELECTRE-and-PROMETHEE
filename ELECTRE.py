import pandas as pd
import numpy as np


class ELECTRE:
    def __init__(self):
        self.data = None
        self.c = None
        self.D = None
        self.C = None
        self.sigma = None
        self.ranking_descending = {}

    
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
                f = np.where(self.D[i, j, :] > self.C[i, j])
                if len(f[0]) == 0:
                    struct[i, j] = self.C[i, j]
                else:
                    if np.any(self.D[i, j, f] == 1):
                        struct[i, j] = 0
                    else:
                        struct[i, j] = self.C[i, j] * np.prod((1 - self.D[i, j, f]) / (1 - self.C[i, j]))

        self.sigma = struct

    
    def destilation_descending(self):

        s = lambda x: -0.15 * x + 0.3

        indices = np.arange(0, self.sigma.shape[0])
        np.fill_diagonal(self.sigma, 0)
        
        place = 1


        while self.sigma.size != 0:
            # 1, 2
            lambda_upper = np.max(self.sigma)

            # 4
            if lambda_upper == 0:
                break

            # 3
            lambda_lower = np.max(self.sigma[
                np.where(self.sigma < lambda_upper - s(lambda_upper))
            ])

            # 5
            checklist = np.where((self.sigma > lambda_lower) & (self.sigma > self.sigma.T + s(self.sigma)), self.sigma, 0)

            # 6
            strength = np.count_nonzero(checklist, axis=1)
            weakness = np.count_nonzero(checklist, axis=0)
            quality = strength - weakness

            best_alternatives = np.where(quality == np.max(quality))[0]
            idxs = indices[best_alternatives]
        
            # 8, 9
            while best_alternatives.size != 1:
                if np.all(checklist == 0):
                    break

                initial_destilation = self.sigma[best_alternatives]
                initial_destilation = initial_destilation[:, best_alternatives]

                lambda_upper = lambda_lower
                lambda_lower = np.max(initial_destilation[
                    np.where(initial_destilation < lambda_upper - s(lambda_upper))
                ])

                checklist = np.where((initial_destilation > lambda_lower) & (initial_destilation > initial_destilation.T + s(initial_destilation)), initial_destilation, 0)

                strength = np.count_nonzero(initial_destilation, axis=0)
                weakness = np.count_nonzero(initial_destilation, axis=1)
                quality = strength - weakness

                best_alternatives = np.where(quality == np.max(quality))[0]
                idxs = idxs[best_alternatives]

            # 7
            self.ranking_descending[place] = idxs

            for i in idxs:
                indices = indices[indices != i]

            if indices.size != 0:
                for i in range(2):
                    self.sigma = np.delete(self.sigma, self.ranking_descending[place], axis=i)
            else:
                self.sigma = np.array([])
            place += 1

            print(self.ranking_descending)