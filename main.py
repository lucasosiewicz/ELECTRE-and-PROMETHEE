from ELECTRE import ELECTRE

P = [200, 3, 1000, 1000]
Q = [100, 1, 500, 500]
V = [300, 6, 1500, 1500]
WEIGHTS = [4, 4, 2, 3]


def main(filename):
    electre = ELECTRE()
    electre.load_data(filename)
    electre.count_marginal_concordace(P, Q)
    electre.count_marginal_discordace(P, V)
    electre.count_total_concordance(WEIGHTS)
    electre.count_outranking_credibility()
    electre.destilation(descending=True)
    electre.destilation(descending=False)
    electre.reverse_ranking()
    electre.create_median_ranking()
    print(electre.median_ranking)




if __name__ == '__main__':
    main('data.csv')