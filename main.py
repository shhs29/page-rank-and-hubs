import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def unscaled_page_rank(N_matrix):
    print("Unscaled Page Rank")

    r_vector = np.full((18, 1), 1 / 18)
    k = 1
    iter_1 = np.linalg.matrix_power(N_matrix.transpose(), k)
    k = k + 1
    iter_2 = np.linalg.matrix_power(N_matrix.transpose(), k)
    k = k + 1
    prev_page_rank = np.matmul(iter_1, r_vector)
    curr_page_rank = np.matmul(iter_2, r_vector)
    while not np.allclose(prev_page_rank, curr_page_rank):
        iter_1 = np.linalg.matrix_power(N_matrix.transpose(), k)
        k = k + 1
        prev_page_rank = curr_page_rank
        curr_page_rank = np.matmul(iter_1, r_vector)
    print(k-1)
    print(np.sum(curr_page_rank))
    print(curr_page_rank)


def scaled_page_rank(N_hat_matrix):
    print("Scaled Page Rank")

    r_vector = np.full((18, 1), 1 / 18)
    k = 1
    iter_1 = np.linalg.matrix_power(N_hat_matrix.transpose(), k)
    k = k + 1
    iter_2 = np.linalg.matrix_power(N_hat_matrix.transpose(), k)
    k = k + 1
    prev_page_rank = np.matmul(iter_1, r_vector)
    curr_page_rank = np.matmul(iter_2, r_vector)
    while not np.allclose(prev_page_rank, curr_page_rank):
        iter_1 = np.linalg.matrix_power(N_hat_matrix.transpose(), k)
        k = k + 1
        prev_page_rank = curr_page_rank
        curr_page_rank = np.matmul(iter_1, r_vector)
    print(k - 1)
    print(np.sum(curr_page_rank))
    print(curr_page_rank)


def page_rank(s):
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
    plt.figure(1, figsize=(8, 8))
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
    new_labels = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
                  15: 16, 16: 17, 17: 18}
    nx.draw(G, labels=new_labels, with_labels=True)
    plt.show()

    N_matrix = adj_matrix.astype(float)
    N_hat_matrix = copy.deepcopy(N_matrix)
    for idx in range(len(N_matrix)):
        row_sum = np.sum(N_matrix[idx])
        divide = np.array(row_sum)
        if row_sum == 0:
            N_matrix[idx][idx] = 1
        else:
            N_matrix[idx] = N_matrix[idx] / divide
            N_matrix[np.isnan(N_matrix)] = 0
        N_hat_matrix[idx] = N_matrix[idx]
        # print(N_matrix[idx])
        N_hat_matrix[idx] = (s * N_hat_matrix[idx]) + ((1 - s) / 18)
        # print(N_hat_matrix[idx])
    unscaled_page_rank(N_matrix)
    scaled_page_rank(N_hat_matrix)


if __name__ == '__main__':
    page_rank(0.9)
