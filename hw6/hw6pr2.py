"""
Starter file for hw6pr2 of Big Data Summer 2017

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
    1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import urllib
import PIL

if __name__ == '__main__':

    # =============STEP 0: LOADING DATA=================
    # NOTE: Be sure to install Pillow with "pip3 install Pillow"
    print('==> Loading image data...')
    img = np.array(PIL.Image.open(urllib.request.urlopen('http://i.imgur.com/X017qGH.jpg')))

    #convert to grayscale because that's what we display at the end anyway
    #(0.3 * R) + (0.59 * G) + (0.11 * B) 
    img = 0.3*img[:,:,0] + 0.59*img[:,:,1] + 0.11*img[:,:,2]
    
    # Shuffle the image

    # HINT:
        # 1) Use np.random.shuffle(img) to shuffle an image img
        # 2) np.random.shuffle() only shuffle along the major axis (row).
        #     Be sure to flatten the image with img.flatten() before doing the shuffling

    "*** YOUR CODE HERE ***"
    shuffle_img = img.copy().flatten()
    np.random.shuffle(shuffle_img)


    "*** END YOUR CODE HERE ***"
    # reshape the shuffled image
    shuffle_img = shuffle_img.reshape(img.shape)

    # =============STEP 1: RUNNING SVD ON IMAGES=================
    print('==> Running SVD on images...')

    # TODO: SVD on img and shuffle_img

    # HINT:
    #         1) Use np.linalg.svd() to perform singular value decomposition
    #         2) For the naming of variables, decompose img into U, S, V
    #         3) Decompose shuffle_img into U_s, S_s, V_s

    "*** YOUR CODE HERE ***"
    
    U, S, V = np.linalg.svd(img)
    
    U_s, S_s, V_s = np.linalg.svd(shuffle_img)


    "*** END YOUR CODE HERE ***"

    # =============STEP 2: SINGULAR VALUE DROPOFF=================
    print('==> Singular value dropoff plot...')
    k = 100
    plt.style.use('ggplot')
    # TODO: Generate singular value dropoff plot

    # NOTE:
    #         1) Make sure to generate lines with different colors or markers

    "*** YOUR CODE HERE ***"
    plt.figure()
    orig_S_plot = plt.plot(S[:k], c = 'b')
    shuf_S_plot = plt.plot(S_s[:k], c = 'r')


    "*** END YOUR CODE HERE ***"

    plt.legend((orig_S_plot, shuf_S_plot), \
        ('original', 'shuffled'), loc = 'best')
    plt.title('Singular Value Dropoff for Clown Image')
    plt.ylabel('singular values')
    plt.savefig('dropoff.png', format='png')
    plt.close()

    # =============STEP 3: RECONSTRUCTION=================
    print('==> Reconstruction with different ranks...')
    rank_list = [2, 10, 20]
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='Greys_r')
    plt.axis('off')
    plt.title('Original Image')

    # Generate reconstruction images for each of the rank values

    # HINT:
    #         1) Use plt.imshow() to display images
    #         2) Set cmap='Greys_r' in imshow() to display grey scale images

    for index in range(len(rank_list)):
        k = rank_list[index]
        plt.subplot(2, 2, 2 + index)

        "*** YOUR CODE HERE ***"
        u_cutoff = U[:,:k]
        s_cutoff = np.diag(S[:k])
        v_cutoff = V[:k,:]
        img_reconstruct = u_cutoff@s_cutoff@v_cutoff
        plt.imshow(img_reconstruct,cmap='Greys_r')
        


        "*** END YOUR CODE HERE ***"

        plt.title('Rank {} Approximation'.format(k))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction.png', format='png')
    plt.close()
