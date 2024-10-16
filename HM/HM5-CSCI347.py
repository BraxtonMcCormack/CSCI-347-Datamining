import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN



def q1():
    #generate 2D matrix data
    mu = np.array([0,0])
    Sigma = np.array([[1,0],[0,1]])
    X1, X2 = np.random.multivariate_normal(mu, Sigma, 1000).T
    D = np.array([X1.T, X2.T])
    plt.scatter(D[0],D[1])
    plt.show()
    return D

def q2(D):
    #setup matrices
    t = (m.pi/4)
    R = np.array([[m.cos(t), -m.sin(t)],[m.sin(t), m.cos(t)]]) #rotation
    S = np.array([[5,0],[0,2]]) #Scale
    
    #dot products
    RS = np.dot(R, S)   
    RSD = np.dot(RS, D)
    print(f"RS: \n{RS}")
    print(f"RDS: \n{RSD}")

    #2a: scatter plot of transformed data
    print("Part 2a: Plotted RSD against D")
    plt.scatter(RSD[0], RSD[1], c="r", alpha=.5)
    plt.scatter(D[0], D[1], c= "b", alpha=.5)
    plt.legend(["RSD", "OG Data"])
    plt.show()

    #2B:Covariance
    print(f"\n2b: Covariance of RS Data")
    print(np.cov(RSD))

    #2C: Total Variance
    print("\n2c: Total Variance on RS Data")
    print(np.var(RSD))
    return RSD

def q3(RSD):
    print("\n3: PCA")
    pca = PCA(n_components=2) #2d
    pca.fit(RSD)    #fits data to PCA with 2 components
    
    print("\n3a: plot PCAs ")
    plt.scatter(pca.components_[0], pca.components_[1])
    plt.show()

    print("\n3b: Sample covariance ")
    print(np.cov(pca.components_))
    
    print("\n3c: ")
    print(f"PCA1's Variance ratio: {pca.explained_variance_ratio_[0]}")
    print(f"PCA2's Variance ratio: {pca.explained_variance_ratio_[1]}")
    

def q4():
    # load dataset
    ds = load_data('Boston')
    print("Question 4.1 (plot)")
    # 4.1 Standardize/normalize dataset
    boston = StandardScaler().fit_transform(ds)
    boston = pd.DataFrame(boston)
    # Make PCA with 2 dimensions
    boston2D = PCA(n_components=2)
    ds2D = boston2D.fit_transform(boston)
    print(ds2D)
    # Plot data
    plt.scatter(ds2D[:,0], ds2D[:,1])
    plt.title("Boston PCA Scatterplot")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    # 4.2 Fraction of total variance plot
    print(f"\nQuestion 4.2: Fraction of Total variance (plot)")
    boston13D = PCA(n_components=13).fit(boston) # 13 dimensions
    # plots data
    ratioSum = boston13D.explained_variance_ratio_.cumsum()
    plt.bar(range(1,len(ratioSum)+1), boston13D.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1,len(ratioSum)+1), ratioSum, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.ylim (0,1.05)
    plt.xticks(range(1,14))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Principal component')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    
    print("\nQuestion 4.3.1")
    print("For 90% capture, we need 7 Principle Components")

    print("\nQuestion 4.3.2")
    print(f"61% of variance is captured through the first two Principle Components")
    
    print("\nQuestion 4.4")
    kmeans = KMeans(n_clusters=3, init='k-means++')
    pred_cluster_labels = kmeans.fit_predict(ds2D)

    plt.scatter(ds2D[:,0],ds2D[:,1], c=pred_cluster_labels)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', s=50)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    boston2D = PCA(n_components=2)
    ds2D = boston2D.fit_transform(boston)
    print("\nQuestion 4.5")
    db = DBSCAN().fit(ds2D)
    labels = db.labels_
    plt.scatter(ds2D[:, 0], ds2D[:, 1], c=labels, cmap='viridis', s=25, alpha=0.8)
    plt.title("Boston DBSCAN plt")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

def main():
    D = q1()
    RSD = q2(D)
    q3(RSD)
    q4()

main()