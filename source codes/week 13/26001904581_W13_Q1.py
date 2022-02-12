# Tian Xiaoyang
# 26001904581
# self built k means clustering classification

# import necessary libraries
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt

# the main k means class
class K_means:
    def __init__(self, X, K):# initilize arguments and functions
        self.X = X
        self.Output = {}
        self.Centroids = np.array([]).reshape(self.X.shape[1], 0) # reshape the centroid array to fit the input data
        self.K = K
        self.m = self.X.shape[0]
    # the k means prediction function
    def kmeans_predict(self, X_data, K_number):
        i = rd.randint(0, X_data.shape[0]) # random integer as the index in the data, to get random centroids
        init_centroid = np.array([X_data[i]]) # initial centroid is a random array based on input data
        for k in range(1, K_number):
            D = np.array([]) # create en emoty list
            for x in X_data:
                D = np.append(D, np.min(np.sum((x - init_centroid)**2))) # within the range of data, add minimum of the sum of squares of differences between centroids and x
            probs = D/np.sum(D) # calcualte probabilities
            cummulative_probs = np.cumsum(probs) # cumulative probabiities
            r = rd.random()
            i = 0
            for a, b in enumerate(cummulative_probs):
                if r < b:
                    i = a
                    break
            init_centroid = np.append(init_centroid, [X_data[i]], axis = 0)
        return init_centroid.T # return the transposed centroids array
    
    def fit(self, no_iter): # function to fit the data points to the centroids
        #randomly initialize the centroids
        self.Centroids = self.kmeans_predict(self.X, self.K)
        
        #compute euclidian distances and assign clusters
        for n in range(no_iter): 
            Distance = np.array([]).reshape(self.m, 0)
            for k in range(self.K):
                temp_dist = np.sum((self.X-self.Centroids[:, k])**2, axis = 1)
                Distance = np.c_[Distance, temp_dist]
            Cents = np.argmin(Distance, axis=1)+1
            #update the centroids
            cents_update = {}
            for k in range(self.K): # withint centroids range
                cents_update[k+1] = np.array([]).reshape(2,0) # update the centroids
            for i in range(self.m):
                cents_update[Cents[i]] = np.c_[cents_update[Cents[i]], self.X[i]]
        
            for k in range(self.K):
                cents_update[k+1] = cents_update[k+1].T
            for k in range(self.K):
                self.Centroids[:, k]= np.mean(cents_update[k+1],axis=0)
                
            self.Output = cents_update
            
    def predict(self):# predicttion function, returns prediction output and centroids
        return self.Output, self.Centroids.T

if __name__ == "__main": # enter variables and plot the graphs
    dataset = pd.read_csv('/Users/wcwe/Desktop/Data science/13/q1.csv')
    X = dataset.iloc[:, [1, 3]].values
    m = X.shape[0]
    n_iter = 100 # 100 iterations
    K = 3 # 3 centroids
    kmeans = K_means(X,K) # feed the data and iterations into the k means function
    kmeans.fit(n_iter) # fit the model to data
    Output, Centroids = kmeans.predict() # preditc the 
    color=['red','blue','green'] # color for different clusters
    labels=['cluster1','cluster2','cluster3']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(Centroids[:,0],Centroids[:,1],s=300,c='yellow',label='Centroids')
    plt.title('variable clusters')
    plt.xlabel('individual') # x llabel
    plt.ylabel('variables') # y label
    plt.legend()# show legends
    plt.show() # show the graphs