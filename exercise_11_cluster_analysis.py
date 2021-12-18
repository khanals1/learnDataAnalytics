import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster


def clustering():
    customerdf = pd.read_csv("PotentialClients.csv")
    customerdf = customerdf.astype('float64')
    plt.plot(customerdf, 'x')
    plt.title("Raw Data")
    plt.show()

    ini = [cluster.vq.kmeans(customerdf, i) for i in range(1, 25)]
    plt.title("Elbow Plot")
    plt.plot([var for (cent, var) in ini]) 
    plt.show()

    cent, var = ini[7]
    assignment, cdist = cluster.vq.vq(customerdf, cent)
    plt.scatter(customerdf.iloc[:, 0], customerdf.iloc[:, 1], c=assignment)
    plt.title("Partioned Cluster")
    plt.show()


if __name__ == '__main__':
    clustering()
