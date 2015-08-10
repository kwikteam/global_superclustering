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

def make_sythentic_categorical_data(tiling_int, num_kkruns,num_global_clusters, max_local_clusters, num_starting_clusters): 
    pretiledata = np.random.randint(max_local_clusters, size=(num_global_clusters,num_kkruns))
#In [4]: np.random.randint(5, size=(6,4))
#Out[4]: 
#array([[2, 0, 2, 1],
#       [1, 4, 0, 1],
#       [1, 2, 2, 2],
#       [0, 2, 2, 4],
#       [4, 1, 4, 3],
#       [4, 1, 2, 3]])
    
    data =  np.tile(pretiledata, (tiling_int,1))
#array([[3, 4, 0, 3],
       #[3, 3, 1, 1],
       #[1, 0, 1, 1],
       #[0, 4, 3, 1],
       #[3, 2, 1, 1],
       #[1, 2, 1, 0],
       #[3, 4, 0, 3],
       #[3, 3, 1, 1],
       #[1, 0, 1, 1],
       #[0, 4, 3, 1],
       #[3, 2, 1, 1],
       #[1, 2, 1, 0],
       #[3, 4, 0, 3],
       #[3, 3, 1, 1],
       #[1, 0, 1, 1],
       #[0, 4, 3, 1],
       #[3, 2, 1, 1],
       #[1, 2, 1, 0]])
    num_spikes = data.shape[0]#tiling_factor*num_global_clusters
    #assert num_spikes = tiling_factor*num_global_clusters
    initial_clustering = np.random.randint(num_starting_clusters, size = num_spikes)   
    return data, initial_clustering
       
if __name__ == "__main__":
    tiling_int = 1000
    num_kkruns = 30
    num_global_clusters = 14
    max_local_clusters = 10
    num_starting_clusters = 14
    [test_superclusters, init_clust] = make_sythentic_categorical_data(tiling_int,\
           num_kkruns,num_global_clusters, max_local_clusters, num_starting_clusters)
    
    sparsedata = sorting.sparsify_superclusters(test_superclusters)
    outsil = sparsedata.to_sparse_data() #don't need to write the outsil variable, everything is stored within the sparse class
    distdata = sparsedata.supercluster_distribution()
    #start_time = time.time()
    #[clust50, dic50] = superdata.clump_fine_clustering(50)
    #time_taken_clump = time.time()-start_time
    #print('Time taken for clump clustering %.2f s' %(time_taken_clump))
    #embed()
    script_params = default_parameters.copy()
   
    script_params.update(
        consider_cluster_deletion = False,
        
        )
    
    
    kk = KK(sparsedata,**script_params)
    kk.cluster_from(init_clust)