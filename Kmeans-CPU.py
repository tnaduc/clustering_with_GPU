
# coding: utf-8

# In[12]:

import math
import numpy as np
import itertools
from matplotlib import pyplot as plt
import tensorflow as tf
import time

nb_clusters = 3
nb_samples = 100000


true_centers = np.random.uniform (-100,100, (nb_clusters,2))
cov = np.diag([8.0,8.0])
samples =  np.concatenate([np.random.multivariate_normal(true_centers[i], cov, nb_samples) 
                              for i in range(nb_clusters)])
true_labels = np.concatenate([np.ones(nb_samples)*i for i in range(nb_clusters)]).reshape((1,nb_samples*nb_clusters))

def plot_clusters(centers, samples, labels, size = 5):
    plt.figure(figsize=(12,8))
    plt.scatter(samples[:,0],samples[:,1], c=labels, s= size)
    nb_samples_per_center = int(len(samples)/len(centers))
    for i, center in enumerate(centers):
        samples_in_one_center = samples[i*nb_samples_per_center:(i+1)*nb_samples_per_center]
        plt.plot(center[0],center[1], markersize =1, marker ='x', color ='k', mew =25)
        plt.plot(center[0],center[1], markersize =1, marker ='o', color ='k', mew =25)
        
        
        
#################################### k-means ####################################        
        
        
def random_initialize_centroids(data, nb_centroids):
    samples = np.copy(data)
    np.random.shuffle(samples)
    return samples[:nb_centroids]

def nearest_centroid(data, centroids):
    distances = np.sqrt(((data - np.expand_dims(centroids,1))**2).sum(axis=2))
    labels = np.argmin(distances, axis = 0)
    return labels

def move_to_new_centroids(data, labels, centroids):
    unique_labels = range(centroids.shape[0])
    new_centroids = np.array([data[labels ==label].mean(axis=0) for label in unique_labels])
    return  new_centroids   


def k_means(data, nb_centroids, iteration = 1000):
    centroids = random_initialize_centroids(data, nb_centroids)
    for i in range(iteration):
        labels = nearest_centroid(data, centroids)
        new_centroids = move_to_new_centroids(data, labels, centroids)
        centroids = new_centroids
    return centroids, labels    



start = time.time()
est_centroids, est_labels = k_means(samples, 3,2)
end =time.time()
print(end-start)


# In[ ]:




# In[ ]:



