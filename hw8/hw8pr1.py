"""
Starter file for k-means(hw8pr1) of Big Data Summer 2017

The file is seperated into two parts:
    1) the helper functions
    2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
    1) Let m be the number of samples
    2) Let n be the number of features
    3) Let k be the number of clusters

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

#########################################
#             Helper Functions            #
#########################################

def k_means(X, k, eps=1e-10, max_iter=1000, print_freq=25):
    """    This function takes in the following arguments:
            1) X, the data matrix with dimension m x n
            2) k, the number of clusters
            3) eps, the threshold of the norm of the change in clusters
            4) max_iter, the maximum number of iterations
            5) print_freq, the frequency of printing the report

        This function returns the following:
            1) clusters, a list of clusters with dimension k x 1
            2) label, the label of cluster for each data with dimension m x 1
            3) cost_list, a list of costs at each iteration

        HINT:
            1) Use np.argsort to get the index of the sorted array
            2) Use copy.deepcopy(A) to make a deep copy of an object A
            3) Calculate cost with k_means_cost()
            2) Each iteration consists of two steps:
                a) finding the closest center for each data point
                b) update the centers according to the data points

        NOTE:
            1) We use l2-norm as the distance metric
    """
    m, n = X.shape
    cost_list = []
    t_start = time.time()
    # randomly generate k clusters
    clusters = [np.random.multivariate_normal((.5 + np.random.rand(n)) * X.mean(axis=0), 10 * X.std(axis=0) * np.eye(n)) for clust in range(k)]
    clusters = np.array(clusters)
    label = np.zeros((m, 1)).astype(int)
    iter_num = 0

    while iter_num < max_iter:

        # Implement k-means algorithm
        "*** YOUR CODE HERE ***"
        #2) calculate distance between each data point and cluster centers
        distances = calc_sq_distance(X,clusters)
        
        #3) Assign each data point to the closest cluster center
        label = np.apply_along_axis(lambda x:np.argmin(x), axis=1, arr=distances)
        
        #4) update the cluster centers
        prev_clusters = clusters.copy()
        clusters = calc_means(X,label,k)

        "*** END YOUR CODE HERE ***"

        # Calculate cost and append to cost_list
        "*** YOUR CODE HERE ***"
        
        cost = k_means_cost(X, clusters, label)
        cost_list.append(cost)

        "*** END YOUR CODE HERE ***"

        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
        if np.linalg.norm(prev_clusters - clusters) <= eps:
            print('-- Algorithm converges at iteration {} \
                with cost {:4.4E}'.format(iter_num + 1, cost))
            break
        iter_num += 1

    t_end = time.time()
    print('-- Time elapsed: {t:2.2f} \
        seconds'.format(t=t_end - t_start))
    return clusters, label, cost_list

def calc_sq_distance(X, clusters):
    #calculate the squared distance between each point and each cluster mean
    k = clusters.shape[0]
    distances = np.zeros((X.shape[0],k))
    for cluster_i in range(k):
        distances[:,cluster_i] = np.linalg.norm(X-clusters[cluster_i], axis=1)**2
    return distances
    
    
def calc_means(X, label, k):
    
    #calculate the new means of each cluster
    m,n = X.shape
    new_means = np.zeros((k,n))
    for cluster_i in range(k):
        in_cluster = label == cluster_i
        new_means[cluster_i] = np.mean(X[in_cluster], axis=0)
    return new_means
        
        
        

def k_means_cost(X, clusters, label):
    """    This function takes in the following arguments:
            1) X, the data matrix with dimension m x n
            2) clusters, the matrix with dimension k x n
            3) label, the label of the cluster for each data point with
                dimension m x 1

        This function calculates and returns the cost for the given data
        and clusters.

        NOTE:
            1) The total cost is defined by the sum of the l2-norm difference
            between each data point and the cluster center assigned to this data point
    """
    m, n = X.shape
    k = clusters.shape[0]

    # Calculate the total cost
    "*** YOUR CODE HERE ***"
    cost = np.sum(np.linalg.norm(X-clusters[label], axis=1)**2)
    
    #calculate the cluster means
    
    

    "*** END YOUR CODE HERE ***"
    return cost


###########################################
#            Main Driver Function             #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':
    # =============STEP 0: LOADING DATA=================
    print('==> Step 0: Loading data...')
    # Read data
    path = '5000_points.csv'
    columns = ['x', 'space', 'y']
    features = ['x', 'y']
    df = pd.read_csv(path, sep='  ', names = columns, engine='python')
    X = np.array(df[:][features]).astype(int)


    # =============STEP 1a: Implementing K-MEANS=================
    # Fill in the code in k_means() and k_means_cost()
    # NOTE: You may test your implementations by running k_means(X, k)
    #         for any reasonable value of k

    # =============STEP 1b: FIND OPTIMAL NUMBER OF CLUSTERS=================
    # Calculate the cost for k between 1 and 20 and find the k with
    #         optimal cost
    print('==> Step 1: Finding optimal number of clusters...')
    cost_k_list = []
    for k in range(1, 21):
        # Get the clusters, labels, and cost list for different k values
        "*** YOUR CODE HERE ***"
        clusters, label, cost_list = k_means(X, k)
        
        "*** END YOUR CODE HERE ***"
        cost = cost_list[-1]
        cost_k_list.append(cost)
        print('-- Number of clusters: {} - cost: {:.4E}'.format(k, cost))

    opt_k = np.argmin(cost_k_list) + 1
    print('-- Optimal number of clusters is {}'.format(opt_k))

    # Generate plot of cost vs k
    "*** YOUR CODE HERE ***"
    plt.figure()
    plt.plot(np.arange(1,21),cost_k_list)
    plt.xlabel('number of clusters (k)')
    plt.ylabel('objective function in final iteration')

    "*** END YOUR CODE HERE ***"

    plt.title('Cost vs Number of Clusters')
    plt.savefig('kmeans_1.png', format='png')
    plt.close()

    # =============STEP 1c: VISUALIZATION=================
    # Generate visualization on running k-means on the optimal k value
    # NOTE: Be sure to mark the cluster centers from the data point
    "*** YOUR CODE HERE ***"
    
    #reget the clusters
    clusters, label, cost_list = k_means(X, opt_k)
    
    #plot each cluster seperately
    plt.figure()
    data_plot = []
    for cluster_i in range(opt_k):
        data_plot.append(plt.scatter(X[label==cluster_i,0], X[label==cluster_i,1], marker='.'))
    
    #plot all the cluster means in red
    center_plot = []
    for cluster_i in range(opt_k):
        center_plot.append(plt.scatter(clusters[cluster_i,0], clusters[cluster_i,1], marker='x', c='k'))

    "*** END YOUR CODE HERE ***"

    # set up legend and save the plot to the current folder
    plt.title('Visualization with {} clusters'.format(opt_k))
    plt.savefig('kmeans_2.png', format='png')
    plt.close()
