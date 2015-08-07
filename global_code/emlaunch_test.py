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


script_params = default_parameters.copy()
#script_params.update(
#        run_monitoring_server=False,
#        debug=True,
#        )

picklefile = '/home/skadir/globalphy/nicktest/nick_global_80001_supercluster.p'
pkl_file = open(picklefile,'rb')
[time_taken_parallel, full_adjacency, channel_order_dict,globalcl_dict,supercluster_info,supercluster_results, superclusters] = pickle.load(pkl_file)
pkl_file.close()  

#superdata used to be called silly
superdata = sorting.sparsify_superclusters(superclusters)
outsil = superdata.to_sparse_data() #don't need to write the outsil variable, everything is stored within the sparse class
distdata = superdata.supercluster_distribution()
start_time = time.time()
[clust50, dic50] = superdata.clump_fine_clustering(50)
time_taken_clump = time.time()-start_time
print('Time taken for clump clustering %.2f s' %(time_taken_clump))
embed()
kk = KK(superdata,**script_params)
kk.cluster_from(clust50)

#Automatically create clust100 via
#kk.cluster_hammingmask_starts(100)
embed()
