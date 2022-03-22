import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def calculate_hubs_authorities_value(adjacency_matrix, num_iterations):
    print("Calculating hub and authority scores for adjacency matrix")
    initial_hub_vector = np.full((adjacency_matrix.shape[0], 1), 1)
    print("Initial hub vector:")
    print(initial_hub_vector)
    for k in range(num_iterations):
        print(f'Iteration {k + 1}')
        mult1 = np.matmul(adjacency_matrix.transpose(), adjacency_matrix)
        power1 = np.linalg.matrix_power(mult1, k)
        authority_vector = np.matmul(np.matmul(power1, adjacency_matrix.transpose()), initial_hub_vector)
        mult2 = np.matmul(adjacency_matrix, adjacency_matrix.transpose())
        power2 = np.linalg.matrix_power(mult2, k + 1)
        hub_vector = np.matmul(power2, initial_hub_vector)
        print("Authority vector before normalisation")
        print(authority_vector)
        print("Authority vector normalisation value")
        print(np.sum(authority_vector))
        print("Hub vector before normalisation")
        print(hub_vector)
        print("Hub vector normalisation value")
        print(np.sum(hub_vector))


def draw_graph_from(adj_matrix):
    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
    new_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
    nx.draw(G, labels=new_labels, with_labels=True)
    plt.show()


if __name__ == '__main__':
    adj_matrix_q1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0]])
    draw_graph_from(adj_matrix_q1)
    calculate_hubs_authorities_value(adj_matrix_q1, 2)
