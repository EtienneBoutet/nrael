import math
import numpy as np

def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

# Colonnes: la valeur x_i Rangé: nombre d'inputs
X = np.array([[2, 2], [-1, -1]])
# Colonnes: valeur y_i lié à l'input i
Y = np.array([[1], [1]])
n = 2

# Colonnes: Nombre de neuronnes dans couche Rangé: Nombre de poids liés à un neuronne
w1 = np.array([[0.5, -1], [1.5, -2]])
w2 = np.array([[1, -1], [3, -4]])
w3 = np.array([[1], [-3]])

print(X)

print("---")

print(w1)

L2 = sigmoid(np.matmul(X, w1))
# L2 est 2x2 : Colonnes: Valeur du neurone i Rangé: Valeurs lié au input i
# l2 = [[0, 0], [0, 0]]
# for i in range(2):
#     # TODO - Figure out colonnes et rangés
#     for k in range(2):
#         value = 0
#         for j in range(len(w1)):
#             value += x[k][j] * w1[j][k]
#         l2[i][k] = sigmoid(value)

print(L2)

# # L3 est 2x2 : Colonnes: Valeur du neurone i Rangé: Valeurs lié au input i
# l3 = [[0, 0], [0, 0]]
# for i in range(2):
#     # TODO - Figure out colonnes et rangés
#     for k in range(2):
#         value = 0
#         for j in range(len(w2)):
#             value += l2[k][j] * w2[j][k]
#         l3[i][k] = sigmoid(value)

# print(l3)

# # L4 est 2x1 : Colonnes: Valeur du neurone i Rangé: Valeurs lié au input i
# l4 = [[0], [0]]
# for i in range(2):
#     # TODO - Figure out colonnes et rangés
#     for k in range(1):
#         for j in range(len(w3)):
#             value += l3[k][j] * w3[j][k]
#         l4[i][k] = sigmoid(value)