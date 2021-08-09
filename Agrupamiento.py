import numpy
import numpy as np
import matplotlib.pyplot as pl

class Agrupamiento:

    def __init__(self,K=0):
        self.K = K
        self.X = None
        self.centroides = None
        self.elementos_asignados = None

    def fit(self,X):
        self.X = X

    def inicializar_centroides(self):
        #aleatoriamente
        m, n = self.X.shape
        centroides = np.zeros((self.K, n))
        for i in range(self.K):
            centroides[i] = self.X[np.random.randint(0, m+1), :]
        self.centroides = centroides

    def asignar_grupo(self):
        m = self.X.shape[0]
        self.elementos_asignados = np.zeros((m, 1))
        dist = np.zeros((self.K, 1))#distancias al centroide de cada punto
        for i in range(m):
            for k in range(self.K):
                dist[k] = np.sum((self.X[i, :] - self.centroides[k, :])**2)
            self.elementos_asignados[i] = dist.argmin() + 1 #devuelve el indice de la menor distancia del punto

    def computar_centroide(self):
        m, n = self.X.shape
        centroides = np.zeros((self.K, n))
        cont = np.zeros((self.K, 1))
        for i in range(m):
            index = int((self.elementos_asignados[i] - 1)[0])
            centroides[index, :] += self.X[i, :]
            cont[index] += 1
        self.centroides = centroides/cont

    def graficar_1(self):
        m, n = self.X.shape
        color = "rgb"
        for k in range(1, self.K + 1):
            grp = (self.elementos_asignados == k).reshape(m, 1)
            pl.scatter(self.X[grp[:, 0], 0], self.X[grp[:, 0], 1], s=15)
        pl.scatter(self.centroides[:, 0], self.centroides[:, 1], s=120, color="black")
        pl.show()

    def graficar(self,num_iters):
        m, n = self.X.shape
        for i in range(num_iters):
            #colorear los puntos asignados a los respectivos centroides
            color = "rgb"
            for k in range(1,self.K + 1):
                grp = (self.elementos_asignados == k).reshape(m, 1)
                pl.scatter(self.X[grp[:,0], 0], self.X[grp[:,0], 1], s=15, color=color[k-1])
            pl.title("Iteracion " + str(i))
            pl.scatter(self.centroides[:, 0], self.centroides[:, 1], s=120, color="black")
            pl.show()
            self.computar_centroide()
            self.asignar_grupo()

    def graficar2(self,num_iters):
        m, n = self.X.shape
        for i in range(num_iters):
            #colorear los puntos asignados a los respectivos centroides
            color = "rgb"
            for k in range(1,self.K + 1):
                grp = (self.elementos_asignados == k).reshape(m, 1)
                pl.scatter(self.X[grp[:,0], 0], self.X[grp[:,0], 1], s=15)
            pl.title("Iteracion " + str(i))
            pl.scatter(self.centroides[:, 0], self.centroides[:, 1], s=120, color="black")
            pl.show()
            self.computar_centroide()
            self.asignar_grupo()

    def k_means(self,num_iters):
        self.inicializar_centroides()
        for i in range(num_iters):
            self.asignar_grupo()
            self.computar_centroide()
        self.asignar_grupo()

    def rearmar(self):
        m, n = self.X.shape
        X_rearmada = numpy.zeros((m,n))
        for i in range(m):
            grupo_asignado = int(int(self.elementos_asignados[i]) - 1)
            X_rearmada[i] = self.centroides[grupo_asignado]
        return X_rearmada