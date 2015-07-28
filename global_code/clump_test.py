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


little = np.array([[1,4,5,6,0,0,0], [0,0,0,7,7,1,1], [0,0,0,7,7,1,1], [1,4,5,6,0,0,0], [2,0,1,1,1,1,3], [4,5,0,0,0,12,10],
                   [8,8,8,8,4,4,4], [8,8,8,8,4,4,4],[3,0,1,1,56,13,1],[3,0,1,1,56,13,1],
                  [3,0,0,0,0,0,1], [3,0,0,0,0,0,1],[0,0,0,7,7,1,1],[3,0,1,1,56,13,1], [1,4,5,7,0,0,0],[1,4,5,8,0,0,0]], dtype = int)
print(little.shape)
superlittle = sorting.sparsify_superclusters(little)
cute_little = superlittle.to_sparse_data()
cute_distribution = superlittle.supercluster_distribution()

picklefile = '/home/skadir/globalphy/nicktest/nick_global_320001_supercluster.p'
pkl_file = open(picklefile,'rb')
[time_taken_parallel, full_adjacency, channel_order_dict,globalcl_dict,supercluster_info,supercluster_results, superclusters] = pickle.load(pkl_file)
pkl_file.close()  

silly = sorting.sparsify_superclusters(superclusters)
outsil = silly.to_sparse_data()
distdata = silly.supercluster_distribution()
start_time = time.time()
[clust100, dic100] = silly.clump_fine_clustering(100)
time_taken_clump = time.time()-start_time
print('Time taken for clump clustering %.2f s' %(time_taken_clump))
embed()
