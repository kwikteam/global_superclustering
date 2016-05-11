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

personal_homedir = '/Users/shabnamkadir/clustering/'
picklefile = personal_homedir + 'global_superclustering/global_code/synthetic_cat.p'
pkl_file = open(picklefile,'rb')
mixture = pickle.load(pkl_file)
pkl_file.close()  
#embed()
mixture_dict = mixture[0]
num_starting_clusters = 4 #produces an initial random clustering with 4 starting clusters.
num_spikes = mixture_dict['superclusters'].shape[0]
initclust = tc.generate_random_initial_clustering(num_starting_clusters, num_spikes )

saveinit = True
if saveinit:
    with open('init_synthetic_cat_%g.p'%(num_starting_clusters), 'wb') as g:
        pickle.dump(initclust, g)    

exit()
#superdata used to be called silly

