"""
Start file for hw7pr2 of Big Data Summer 2017

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

First, fill in the the code of step 0 in the main driver to load the data, then
please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
    1) Let m be the number of samples
    2) Let n be the number of features

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################
#             Helper Functions            #
#########################################
def find_cost(X, y, W , reg):
    """    This function takes in three arguments:
            1) W, a weight matrix with bias
            2) X, the data with dimension m x (n + 1)
            3) y, the label of the data with dimension m x 1

        This function calculates and returns the l1 regularized
        mean-squared error
    """
    # solve for l1-regularized mse
    "*** YOUR CODE HERE ***"
        
    # #get the squared difference term
    ls_term = np.linalg.norm(X@W-y)**2
    
    #get the regularization term
    reg_term = reg*np.abs(W)
    reg_term[0,:] = 0 #don't regularize the bias term
    reg_term = np.sum(reg_term)
    
    cost = (ls_term + reg_term)/len(y)
    

    # "*** END YOUR CODE HERE ***"
    return cost

def find_grad(X, y, W, reg=0.0):
    """    This function takes in four arguments:
            1) X, the data with dimension m x (n + 1)
            2) y, the label of the data with dimension m x 1
            3) W, a weight matrix with bias
            4) reg, the parameter for regularization

        This function calculates and returns the gradient of W
    """
    # Find the gradient of lasso with respect to W
    "*** YOUR CODE HERE ***"
    #only do the gradient of the least squares part
    #not sure why you divide by X.shape, effectively just changes the learning rate?
    grad = X.T @ (X@W-y)/X.shape[0] 

    "*** END YOUR CODE HERE ***"
    return grad

def prox(X, gamma):
    """ This function takes in two arguments:
            1)  X, a vector
            2) gamma, a scalar

        This function thresholds each entry of X with gamma
        and updates the changes in place.

        Hint:
            1) Use X > gamma to find the index of entries in X
                that are greater than gamma
    """
    # Threshold each entry of X with respect to gamma
    """*** YOUR CODE HERE ***"""
    X[X>gamma]  += -gamma
    X[X<-gamma] += gamma
    X[np.abs(X)<gamma] = 0

    """*** END YOUR CODE HERE ***"""
    return X

def grad_lasso(
    X, y, reg=1e6, lr=1e-12, eps=1e-5,
    max_iter=300, batch_size=256, print_freq=1):
    """ This function takes in the following arguments:
            1) X, the data with dimension m x (n + 1)
            2) y, the label of the data with dimension m x 1
            3) reg, the parameter for regularization
            4) lr, the learning rate
            5) eps, the threshold of the norm for the gradients
            6) max_iter, the maximum number of iterations
            7) batch_size, the size of each batch for gradient descent
            8) print_freq, the frequency of printing the report

        This function returns W, the optimal weight,
        by lasso gradient descent.
    """
    m, n = X.shape
    obj_list = []
    # initialize the weight and its gradient
    W = np.linalg.solve(X.T @ X, X.T @ y)
    W_grad = np.ones((n, 1))

    print('==> Running gradient descent...')
    iter_num = 0
    t_start = time.time()

    # Run gradient descent

    # HINT:
        # 1) Randomly select indices for entries into a batch using
        #     np.random.randint()
        # 2) Find the gradient using the batch
        # 3) Apply the threshold function prox(), according to the problem
        #     statement, to update W
        # 4) Update the cost and append it to obj_list

    while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
        
        
        "*** YOUR CODE HERE ***"
        #select indices for stochastic gradient descent
        inds = np.random.randint(low=0, high=m-1,size=batch_size)
        
        W_grad = find_grad(X[inds], y[inds], W, reg)
        W = prox(W - lr * W_grad, reg * lr) #not sure why reg*lr
        cost = find_cost(X[inds], y[inds], W, reg)
        obj_list.append(cost)



        "*** END YOUR CODE HERE ***"

        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration{} - training cost {: .4f} - \
                sparsity {: .2f}'.format(iter_num + 1, cost, \
                    (np.abs(W) < reg * lr).mean()))
        iter_num += 1

    # Benchmark report
    t_end = time.time()
    print('--Time elapsed for training: {t:4.2f} \
        seconds'.format(t = t_end - t_start))
    return W, obj_list

def lasso_path(X, y, tau_min=1e-8, tau_max=10, num_reg=10):
    """ This function takes in the following arguments:
            1) X, the data with dimension m x (n + 1)
            2) y, the label of the data with dimension m x 1
            3) tau_min, the minimum value for the inverse of regularization parameter
            4) tau_max, the maximum value for the inverse of regularization parameter
            5) num_reg, the number of regularization parameters

        This function returns the list of optimal weights and the corresponding tau values.
    """
    m, n = X.shape
    W = np.zeros((n, num_reg)) # W has the shape n x num_reg
    tau_list = np.linspace(tau_min, tau_max, num_reg)
    for index in range(num_reg):
        reg = 1. / tau_list[index]
        print('--regularization parameter is {:.4E}'.format(reg))

        
        # HINT:
        #     1) Update each column of W to be the optimal weights at
        #         each regularization parameter
        #     2) Use A.flatten() to flatten an array A
        #     3) Set the batch size to be 1024. Leave other default parameters
        #         as they are.
        """*** YOUR CODE HERE ***"""
        
        W_opt, obj_list = grad_lasso(X, y, 1/tau_list[index], batch_size=1024)
        W[:,index] = W_opt.flatten()
        


        """*** END YOUR CODE HERE ***"""

    return W, tau_list






###########################################
#            Main Driver Function             #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':


    # =============STEP 0: LOADING DATA=================
    print('==> Step 0: Loading data...')
    # Read data
    #df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
    #    sep=', ', engine='python')
    df = pd.read_csv('/Users/Eli/Downloads/online_news_popularity.csv', \
        sep=', ', engine='python')
    X = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]]
    y = np.log(df.shares).values.reshape(-1,1)
    X = np.hstack((np.ones_like(y), X))



    # =============STEP 1: LASSO GRADIENT DESCENT=================
    # NOTE: Fill in code in find_cost(), find_grad(), prox() and
    # grad_lasso() for this step
    # We don't require any output for this step, but you may test your
    # implementation by running grad_lasso(X, y)



    # =============STEP 2: LASSO PATH=================
    # NOTE: Fill in code in lasso_path()
    print('==> Step 2: Running lasso path...')
    W, tau_list = lasso_path(X, y, tau_min=1e-15, tau_max=2e-2, num_reg=10)
    # Plotting lasso path
    plt.style.use('ggplot')
    plt.subplot(211)
    lp_plot = plt.plot(tau_list, W.T)
    plt.title('Lasso Path')
    plt.xlabel('$tau = \lambda^{-1}$')
    plt.ylabel('$W_i$')



    # =============STEP 3: FEATURE SELECTION=================
    print('==> Step 3: The most important features are: ')
    # Find the indices for the top five features

    # HINT:
        # 1) Use df.columns to access the feature names
        # 2) Use np.argsort() to sort the values on the first column of W
        #     in step 2

    """*** YOUR CODE HERE ***"""
    names = np.array(df.columns)
    #do the second column because the first column is all 0's (i.e. too big lambda)
    order = np.argsort(np.abs(W[:,0]))[::-1]+1 #plus one so not to include index column
    top_features = names[order][:5]


    """*** END YOUR CODE HERE ***"""
    print(top_features)



    # =============STEP 4: CONVERGENCE PLOT=================
    print('==> Step 4: Generating convergence plot...')
    plt.subplot(212)
    W_reg, obj_list = grad_lasso(X, y, reg=1e5, lr=1e-12, eps=1e-2, max_iter=2500, \
        batch_size=1024, print_freq=250)
    plt.title("Lasso Objective Convergence: $\lambda = 1e5$")
    plt.ylabel("Stochastic Objective")
    plt.xlabel("Iteration")
    plt.plot(obj_list)
    plt.tight_layout()
    plt.savefig('hw7pr2_lasso.png', format = 'png')
    # plt.close()
    print('==> Plotting completed.')
