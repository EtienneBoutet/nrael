# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])
x1 = X[:, 0]
x2 = X[:, 1]

# %%
plt.scatter(x1, x2)
plt.title("x1 vs x2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# %%
coef = np.corrcoef(x1, x2)
print("Le coefficient de corrélation est : {:.2f}".format(coef[0, 1]))

# Il y existe une corrélation forte et positive entre x1 et x2,
# PCA pourra donc nous aider à représenter les données.

# %%
# Centraliser les donnés sur une moyenne à 0
x1 -= np.mean(x1)
x2 -= np.mean(x2)

plt.scatter(x1, x2)
plt.title("x1 vs x2 centré sur une moyenne à 0")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# %%
# Reconstruire X à partir des modifications faites à x1 et x2
X = np.vstack((x1, x2)).T

# %%
# Trouver la matrice de covariance C
C = np.cov(X.T)

# %%
# Trouver les eigenvalues 'w' et les eigenvectors 'v' de C.
w, v = np.linalg.eig(C)

# Trier les eigenvalues et vecteurs
idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

# %%
# À partir des eigenvalues, on peut trouver la variance des eigenvectors,
sample_variance = np.sum(w)
print("Pourcentage de la variance totale de PC1 : {:.2f}\n".format(w[0] / sample_variance))
print("Pourcentage de la variance totale de PC2 : {:.2f}".format(w[1] / sample_variance))


# %%
# Puisque PC1 explique la structure des données on peut choisir
# d'exprimer X par rapport à PC1 seulement.

pc = v[:, 0]
# Array du départ
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])
Y = np.dot(X, pc)

# %%
plt.scatter(Y, None)
plt.xlabel("PC1")
plt.show()