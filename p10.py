import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def create_random_graphs(m, n):
    G = nx.Graph()
    for i, sub_graph_nodes in enumerate(n):
        for j in range(sub_graph_nodes):
            G.add_node((i, j))

        for j in range(sub_graph_nodes):
            for k in range(j + 1, sub_graph_nodes):
                if np.random.binomial(1, m[i][i]):
                    G.add_edge((i, j), (i, k))

    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            for k in range(n[i]):
                for l in range(n[j]):
                    if np.random.binomial(1, m[i][j]):
                        G.add_edge((i, k), (j, l))

    return G


def view(x, y):
    return y * np.dot(x, y) / np.dot(y, y)


def get_cluster(g, pred_clusters):
    # zamień na macierz sąsiedztwa
    laplacian_matrix = np.array(nx.laplacian_matrix(g).todense())
    eig_val, eig_vec = np.linalg.eig(laplacian_matrix)

    new_pos = np.array([view(laplacian_matrix, eig_vec[:, i]) for i in range(pred_clusters)])

    # zrób klasteryzację na eigenvalue
    kmeans = KMeans(n_clusters=pred_clusters)
    kmeans.fit(new_pos.T)
    clusters = kmeans.predict(new_pos.T)

    # zwróć wynik
    return clusters


for no, divisor in zip([3, 4, 5, 3, 4, 5], [5, 5, 5, 2, 2, 2]):
    n = [20 for _ in range(no)]
    m = np.ones((no, no))
    m = m/divisor
    np.fill_diagonal(m, 9)
    m = m/10
    print(m)
    g = create_random_graphs(m, n)
    mapping = get_cluster(g, no)

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'cyan'}
    mapping = [colors[x] for x in mapping]

    plt.figure(figsize=(9, 9))
    options = {
        "node_color": mapping,
        "node_size": 50,
        "linewidths": 0.3,
        "width": 0.3,
        "alpha": 0.5,
        "pos": nx.spring_layout(g)
    }
    nx.draw(g, **options)
    plt.show()
