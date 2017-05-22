
# coding: utf-8

# In[46]:

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



def tf_random_initialization(data, nb_centroids):
    samples = tf.stack(data)  # convert ndarray in Numpy to tensor in Tensorlow
    nb_samples = tf.shape(data)[0]
    random_indices = tf.random_shuffle(tf.range(0,nb_samples))[:nb_centroids]
    random_centroids = tf.gather(samples, random_indices)
    return random_centroids

def tf_nearest_centroids_labels(data, centroids):
    squared_differences = tf.squared_difference(tf.expand_dims(data,0),tf.expand_dims(centroids,1))
    distances = tf.reduce_sum(squared_differences, axis = 2)
    labels = tf.argmin(distances, axis = 0)
    return labels

def tf_move_to_new_centroids(data, labels, nb_centroids):
    samples = tf.stack(data)
    nearest_labels = tf.to_int32(labels)
    clusters = tf.dynamic_partition(samples, nearest_labels, nb_centroids)    
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(cluster, 0),0) for cluster in clusters],0)
    return new_centroids

def tf_k_means(data, nb_centroids, iteration = 1000):
    tf.global_variables_initializer().run()
    initial_centroids =  tf_random_initialization(data, nb_centroids).eval()
    
    for i in range(iteration):
        nearest_centroids_labels = tf_nearest_centroids_labels(data, initial_centroids).eval()
        initial_centroids = tf_move_to_new_centroids(data, nearest_centroids_labels , nb_centroids).eval()

    return initial_centroids, nearest_centroids_labels


start = time.time()
with tf.Session().as_default(): est_cent, est_labs = tf_k_means(samples, nb_centroids = 3,iteration =2)
end =time.time()
print(end-start)


# In[47]:

get_ipython().magic('matplotlib inline')
plot_clusters(est_cent, samples, est_labs)


# In[45]:

tf.reset_default_graph()


# In[ ]:



