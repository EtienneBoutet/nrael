#%%
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%%
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#%%
C = np.cov(X.T)

#%%
w, v = np.linalg.eig(C)

#%%
idx = w.argsort()[::-1]
w = w[idx]
v = v[:, idx]

#%%
sample_variance = np.sum(w)
print("Pourcentage de la variance totale de PC1 : {:.2f}\n".format(w[0] / sample_variance))
print("Pourcentage de la variance totale de PC2 : {:.2f}\n".format(w[1] / sample_variance))
print("Pourcentage de la variance totale de PC3 : {:.2f}\n".format(w[2] / sample_variance))
print("Pourcentage de la variance totale de PC4 : {:.2f}\n".format(w[3] / sample_variance))

#%%
# Drop du dernier PCA car il n'a pas beaucoup d'importance pour représentrer le data
v = v[:, :-1]

#%%
X_prime = np.dot(X, v)

#%%
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_prime[:, 0], X_prime[:, 1], X_prime[:, 2], c=Y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("PC1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("PC2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

#%%
plt.scatter(X_prime[:, 0], X_prime[:, 1], c=Y, cmap=plt.cm.Set1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()