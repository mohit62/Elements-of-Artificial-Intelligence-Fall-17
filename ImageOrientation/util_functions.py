import scipy.spatial.distance as ssd
import numpy as np

# Common functions
def eucl_distance(a, b):

    return ssd.euclidean(a,b)
    #dist=np.linalg.norm(a - b)
    #print dist

# Drop labels
def drop_col(a):
    col= a[:, 0]
    a = np.delete(a, 0, 1)
    a = a.astype(int)

    return [col,a]
