import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import matplotlib.pyplot as plt
import cv2
import argparse
import time
from sklearn.utils.validation import check_array
from scipy.spatial.distance import pdist,squareform
import scipy
import os


parser=argparse.ArgumentParser()
parser.add_argument('--name',type=str,help='image name')
parser.add_argument('--k',type=int,help='k nearest neighbors k')
args=parser.parse_args()


def knn_matting(image, trimap, k,my_lambda=100,sigma=2,k_method='one_norm'):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    ####################################################
    # TODO: find KNN for the given image
    ####################################################
    # get the index of each position
    all_pixs=np.arange(h*w)
    rows,cols=np.unravel_index(all_pixs,(h,w))

    # get feature vector
    feat_1=image.reshape(h*w,c).T
    idx_list=np.vstack((rows,cols))
    feat_2=idx_list/np.sqrt(h*h+w*w)
    feat_vec=np.append(feat_1,feat_2,axis=0).T

    # find k nearest neighbors
    model=sklearn.neighbors.NearestNeighbors(n_neighbors=k,n_jobs=4)
    model.fit(feat_vec)
    knn=model.kneighbors(feat_vec)[1]

    ####################################################
    # TODO: compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    # get affinity matrix A
    row_idx=np.repeat(all_pixs,k)
    col_idx=knn.reshape(h*w*k)
    if k_method=='one_norm':
        kernel=1-np.linalg.norm(feat_vec[row_idx]-feat_vec[col_idx],axis=1)/(c+2)
    elif k_method=='gaussian':
        kernel=np.exp(-(feat_vec[row_idx]-feat_vec[col_idx])**2/sigma**2)
        kernel=np.sum(kernel,axis=1)/(c+2)
    A=scipy.sparse.coo_matrix((kernel,(row_idx,col_idx)),shape=(h*w,h*w))
    
    # get diagonal matrix D and L
    diag_val=np.ravel(A.sum(axis=1))
    D=scipy.sparse.diags(diag_val)
    L=D-A

    # get diagonal matrix M and assigned alpha v
    tri_mark=np.ravel(foreground+background)
    M=scipy.sparse.diags(tri_mark)
    v=np.ravel(foreground)

    # (L+λM)α-λv
    first=2*(L+my_lambda*M)
    second=2*my_lambda*v.T

    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################

    warnings.filterwarnings('error')
    alpha = []
    try:
        sol=scipy.sparse.linalg.spsolve(first,second)
        alpha=np.clip(sol,0,1).reshape(h,w)
    except Warning:
        sol=scipy.sparse.linalg.lsqr(first,second)[0]
        alpha=np.clip(sol,0,1).reshape(h,w)

    return alpha


if __name__ == '__main__':
    start_time=time.time()
    image = cv2.imread(f'./image/{args.name}.png')
    trimap = cv2.imread(f'./trimap/{args.name}.png', cv2.IMREAD_GRAYSCALE)
    background=cv2.imread(f'./background/{args.name}.png')
    result_dir='./result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    alpha = knn_matting(image, trimap,args.k)
    alpha = alpha[:, :, np.newaxis]

    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################

    [h,w,c]=image.shape
    background=cv2.resize(background,(w,h),interpolation=cv2.INTER_AREA)

    result = []
    alpha_3d=np.repeat(alpha,repeats=3,axis=2)
    result=alpha*image+(1-alpha)*background
    cv2.imwrite(f'{result_dir}/{args.name}.png', result)
    print("--- %s seconds ---" % (time.time() - start_time))
