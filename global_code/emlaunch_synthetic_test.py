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

script_params = default_parameters.copy()
#script_params.update(
#        run_monitoring_server=False,
#        debug=True,
#        )

picklefile = '/home/skadir/globalphy/global_superclustering/global_code/synthetic_cat.p'
pkl_file = open(picklefile,'rb')
mixture = pickle.load(pkl_file)
pkl_file.close()  
embed()
mixture_dict = mixture[0]
num_starting_clusters = 8
num_spikes = mixture_dict['superclusters'].shape[0]
initclust = tc.generate_random_initial_clustering(num_starting_clusters, num_spikes )
#superdata used to be called silly
superdata = sorting.sparsify_superclusters(mixture_dict['superclusters'])
outsil = superdata.to_sparse_data() #don't need to write the outsil variable, everything is stored within the sparse class
distdata = superdata.supercluster_distribution()
start_time = time.time()
[clust10, dic10] = superdata.clump_fine_clustering(10)
time_taken_clump = time.time()-start_time
print('Time taken for clump clustering %.2f s' %(time_taken_clump))

kk = KK(superdata,**script_params)
#kk.cluster_from(clust10)
kk.cluster_from(initclust)
#Automatically create clust100 via
#kk.cluster_hammingmask_starts(100)
embed()
