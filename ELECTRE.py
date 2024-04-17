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
        
        # Definicja funkcji S
        s = lambda x: -0.15 * x + 0.3

        # dla ułatwienia implementacji signorowania wartości po przekątnych wypełniam ich wartości zerami
        np.fill_diagonal(self.sigma, 0)

        # do macierzy sigma dołączam indeksy wierszy
        indices = np.arange(0, self.sigma.shape[0], dtype=int)
        self.sigma = np.c_[self.sigma, indices]
        
        # numer iteracji, reprezentuje również miejsce w rankingu
        place = 1


        while self.sigma.size != 0:
            # poszukiwanie górnego progu wiarygodności
            lambda_upper = np.max(self.sigma[:,:-1])
            print(f'max: {lambda_upper}')

            # jeżeli jest równy 0 to kończę wykonywanie algorytmu
            #if lambda_upper == 0:
            #    break

            # poszukiwanie dolnego progu wiarygodności
            lambda_lower = np.max(self.sigma[:,:-1][
                np.where(self.sigma[:,:-1] < lambda_upper - s(lambda_upper))
            ])

            # zapisywanie wartości sigma dla par gdzie A jest preferowane nad B
            checklist = np.where((self.sigma[:,:-1] > lambda_lower) & (self.sigma[:,:-1] > self.sigma[:,:-1].T + s(self.sigma[:,:-1])), self.sigma[:,:-1], 0)

            # obliczam siłę i słabość wariantów oraz ich użyteczność
            strength = np.count_nonzero(checklist, axis=1)
            weakness = np.count_nonzero(checklist, axis=0)
            quality = strength - weakness

            # zapisuję indeksy najlepszych wariantów
            best_alternatives = np.where(quality == np.max(quality))[0]
            idxs = np.array(self.sigma[best_alternatives, -1])
        
            # jeżeli dojdzie do remisu to przechodzę do destylacji wewnętrznej
            while idxs.size != 1:
                initial_destilation = self.sigma[best_alternatives,:-1]
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

                best_alternatives = np.where(quality == np.max(quality))[0]
                idxs = idxs[best_alternatives]

            # przypisuję wariantom miejsce w rankingu
            self.ranking_descending[place] = idxs
            print(f'Place {place}: {self.ranking_descending[place]}')
            print(f'Sigma: {self.sigma}')
            
            # usuwam warianty z macierzy sigma
            for i in idxs:
                indices = indices[indices != i]
            print(f'Indices: {indices}')

            for idx in idxs:
                row_to_delete = np.where(self.sigma[:,-1] == idx)[0]
                for i in range(2):
                    self.sigma = np.delete(self.sigma, row_to_delete, axis=i)

            place += 1
