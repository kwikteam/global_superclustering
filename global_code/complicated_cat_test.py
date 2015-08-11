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

alpha = np.array([1, 1,1, 5, 4,9])
alpha_permie = tc.generate_kkrun_samples_from_permuted_dirichlet(alpha, 6)
print(alpha_permie)
aleph = list(alpha_permie)

bernie = tc.obtain_bernoulli_from_dirichlet_alphalist(aleph)
np.sum(bernie, axis = 1)

superbern = tc.obtain_superclusters_from_bernoulli(bernie, 20)
superbern.shape
print(superbern)