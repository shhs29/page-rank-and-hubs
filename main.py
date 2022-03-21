import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def mult():
    adj_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    N_matrix = adj_matrix.astype(float)
    plt.figure(1, figsize=(8, 8))
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
    new_labels = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
                  15: 16, 16: 17, 17: 18}
    nx.draw(G, labels=new_labels, with_labels=True)
    plt.show()
    for idx in range(len(N_matrix)):
        row_sum = np.sum(N_matrix[idx])
        divide = np.array(row_sum)
        N_matrix[idx] = N_matrix[idx]/divide
        N_matrix[np.isnan(N_matrix)] = 0

    # print(N_matrix)

    r_vector = np.full((18, 1), 1/18)
    # print(r_vector)
    iter_1 = np.linalg.matrix_power(N_matrix.transpose(), 1)
    iter_2 = np.linalg.matrix_power(N_matrix.transpose(), 2)
    iter_3 = np.linalg.matrix_power(N_matrix.transpose(), 3)
    iter_4 = np.linalg.matrix_power(N_matrix.transpose(), 4)
    iter_5 = np.linalg.matrix_power(N_matrix.transpose(), 5)
    page_rank_1 = np.matmul(iter_1, r_vector)
    page_rank_2 = np.matmul(iter_2, r_vector)
    page_rank_3 = np.matmul(iter_3, r_vector)
    page_rank_4 = np.matmul(iter_4, r_vector)
    page_rank_5 = np.matmul(iter_5, r_vector)
    print(page_rank_1)
    print(page_rank_2)
    print(page_rank_3)
    print(page_rank_4)
    print(page_rank_5)


if __name__ == '__main__':
    mult()
