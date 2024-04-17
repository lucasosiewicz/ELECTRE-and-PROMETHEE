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
        self.ranking_ascending = {}

    
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

    
    def destilation(self, descending):
        
        # Definicja funkcji S
        s = lambda x: -0.15 * x + 0.3

        # dla ułatwienia implementacji signorowania wartości po przekątnych wypełniam ich wartości zerami
        np.fill_diagonal(self.sigma, 0)

        # do macierzy sigma dołączam indeksy wierszy
        sigma = self.sigma.copy()
        indices = np.arange(0, self.sigma.shape[0], dtype=int)
        sigma = np.c_[sigma, indices]
        
        # numer iteracji, reprezentuje również miejsce w rankingu
        place = 1


        while sigma.size != 0:
            # poszukiwanie górnego progu wiarygodności
            lambda_upper = np.max(sigma[:,:-1])

            # jeżeli jest równy 0 to kończę wykonywanie algorytmu
            if lambda_upper == 0:
                if descending:
                    self.ranking_descending[place] = sigma[:,-1]
                else:
                    self.ranking_ascending[place] = sigma[:,-1]
                break

            # poszukiwanie dolnego progu wiarygodności
            lambda_lower = np.max(sigma[:,:-1][
                np.where(sigma[:,:-1] < lambda_upper - s(lambda_upper))
            ])

            # zapisywanie wartości sigma dla par gdzie A jest preferowane nad B
            checklist = np.where((sigma[:,:-1] > lambda_lower) & (sigma[:,:-1] > sigma[:,:-1].T + s(sigma[:,:-1])), sigma[:,:-1], 0)

            # obliczam siłę i słabość wariantów oraz ich użyteczność
            strength = np.count_nonzero(checklist, axis=1)
            weakness = np.count_nonzero(checklist, axis=0)
            quality = strength - weakness

            # zapisuję indeksy najlepszych wariantów
            if descending:
                best_alternatives = np.where(quality == np.max(quality))[0]
            else:
                best_alternatives = np.where(quality == np.min(quality))[0]
            idxs = np.array(sigma[best_alternatives, -1])
        
            # jeżeli dojdzie do remisu to przechodzę do destylacji wewnętrznej
            while idxs.size != 1:
                initial_destilation = sigma[best_alternatives,:-1]
                initial_destilation = initial_destilation[:, best_alternatives]

                lambda_upper = lambda_lower

                if lambda_upper == 0:
                    break
                
                try:
                    lambda_lower = np.max(initial_destilation[
                        np.where(initial_destilation < lambda_upper - s(lambda_upper))
                    ])
                except ValueError:
                    lambda_lower = 0

                checklist = np.where((initial_destilation > lambda_lower) & (initial_destilation > initial_destilation.T + s(initial_destilation)), initial_destilation, 0)
                #print(checklist)
                strength = np.count_nonzero(initial_destilation, axis=0)
                weakness = np.count_nonzero(initial_destilation, axis=1)
                quality = strength - weakness

                if descending:
                    best_alternatives = np.where(quality == np.max(quality))[0]
                else:
                    best_alternatives = np.where(quality == np.min(quality))[0]
                idxs = idxs[best_alternatives]

            # przypisuję wariantom miejsce w rankingu
            if descending:
                self.ranking_descending[place] = idxs
            else:
                self.ranking_ascending[place] = idxs
            
            # usuwam warianty z macierzy sigma
            for i in idxs:
                indices = indices[indices != i]
            
            for idx in idxs:
                row_to_delete = np.where(sigma[:,-1] == idx)[0]
                for i in range(2):
                    sigma = np.delete(sigma, row_to_delete, axis=i)

            place += 1


    def reverse_ranking(self):
        keys = list(self.ranking_ascending.keys())
        values = list(self.ranking_ascending.values())
        for k, v in zip(keys[::-1], values):
            self.ranking_ascending[k] = v


