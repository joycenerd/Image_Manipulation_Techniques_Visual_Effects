import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2


def knn_matting(image, trimap, my_lambda=100):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    ####################################################
    # TODO: find KNN for the given image
    ####################################################

    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    
    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################

    warnings.filterwarnings('error')
    alpha = []
    try:
        pass
    except Warning:
        pass

    return alpha


if __name__ == '__main__':

    image = cv2.imread('./image/bear.png')
    trimap = cv2.imread('./trimap/bear.png', cv2.IMREAD_GRAYSCALE)

    alpha = knn_matting(image, trimap)
    alpha = alpha[:, :, np.newaxis]

    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################

    result = []
    cv2.imwrite('./result/bear.png', result)
