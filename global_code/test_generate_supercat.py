#Be sure to run 
#python setup.py build_ext --inplace
#before running this script

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sorting 
from supercluster import *
from klustakwik2 import *
import imp # lets you reload modules using e.g.imp.reload(sorting)
from IPython import embed
import time
from emcat import KK
from default_parameters import default_parameters
import testing_cat as tc


np.set_printoptions(threshold=np.nan)
alpha_list = [[1,1,1,1,25], [2,3,3,3,3,3,3,50], [4,4,4,1,1,1,10], [1,1,1,1,1,1,12], [3,2,2,2,2,100], [1,1,1,1,1,8]]
alpha_mix = [5,5.2,5,1,1,5]
superclust = tc.create_mixture_categorical_data(alpha_list, 1000, 12, alpha_mix = alpha_mix)
print(superclust)

stufftopickle = [superclust]
#print(supercluster_results)
#superinfo = [supercluster_results]
with open('synthetic_cat.p', 'wb') as g:
    pickle.dump(stufftopickle, g)    
embed()