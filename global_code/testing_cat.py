#Be sure to run 
#python setup.py build_ext --inplace
#before running this script

import pickle
import numpy as np
#import matplotlib.pyplot as plt
#import sorting 
#from supercluster import *
#from klustakwik2 import *
import imp # lets you reload modules using e.g.imp.reload(sorting)
from IPython import embed
import time
#from emcat import KK
import itertools
from default_parameters import default_parameters

def generate_random_initial_clustering(num_starting_clusters, num_spikes):
    '''generate an random clustering of num_spikes into num_starting_clusters'''   
    initial_clustering = np.random.randint(num_starting_clusters, size = num_spikes)  
    return initial_clustering

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
  
def generate_kkrun_samples_from_permuted_dirichlet(alpha, num_kkruns):
    '''
    INPUT:
    alpha = np.array([1, 1, 1, 1, 1, 5, 4, 9])
    num_kkruns = 6
    
    OUTPUT:
    alpha_permie_list = generate_kkrun_samples_from_permuted_dirichlet(alpha, 6)
    [array([ 4.,  1.,  1.,  9.,  1.,  1.,  1.,  5.]),
     array([ 4.,  5.,  1.,  1.,  1.,  1.,  1.,  9.]), 
     array([ 1.,  1.,  9.,  4.,  1.,  5.,  1.,  1.]), 
     array([ 1.,  4.,  1.,  1.,  9.,  1.,  5.,  1.]), 
     array([ 9.,  1.,  5.,  1.,  1.,  1.,  1.,  4.]), 
     array([ 5.,  9.,  1.,  1.,  4.,  1.,  1.,  1.])]
    
    
    '''
    alpha_perms = np.zeros((num_kkruns,len(alpha)))
    for k in np.arange(num_kkruns):
        alpha_perms[k,:] = np.random.permutation(alpha)
    alpha_perms_list = list(alpha_perms)    
    return alpha_perms_list    
    
def obtain_bernoulli_from_dirichlet_alphalist(alpha_list):
    '''
    INPUT:
    alpha_list is a list of arrays of positive values
    each element corresponds to one local kk run
    for each element return one sample from its Dirichlet distribution
    
    [array([ 4.,  1.,  1.,  9.,  1.,  1.,  1.,  5.]),
     array([ 4.,  5.,  1.,  1.,  1.,  1.,  1.,  9.]), 
     array([ 1.,  1.,  9.,  4.,  1.,  5.,  1.,  1.]), 
     array([ 1.,  4.,  1.,  1.,  9.,  1.,  5.,  1.]), 
     array([ 9.,  1.,  5.,  1.,  1.,  1.,  1.,  4.]), 
     array([ 5.,  9.,  1.,  1.,  4.,  1.,  1.,  1.])]
    
    
    OUTPUT:
    bernoulli matrix for a global cluster
    shape = (num_kkruns, num_categories_local clustering)
    
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    
    [[ 0.18005132  0.00899087  0.08869126  0.41377366  0.05540564  0.00188961  0.08648227  0.16471536]
     [ 0.25082722  0.12359689  0.02112979  0.11748895  0.00217658  0.02880718  0.0214635   0.4345099 ]
     [ 0.01233977  0.02088034  0.4955683   0.13625201  0.01145464  0.27812387  0.04226948  0.00311158]
     [ 0.06647051  0.13511664  0.00933061  0.06188849  0.45395058  0.0559436   0.13988709  0.07741248]
     [ 0.32872549  0.03088037  0.17917781  0.04121222  0.01027417  0.19813323  0.04873399  0.16286271]
     [ 0.23774698  0.4250537   0.05125592  0.10071934  0.13196959  0.00447368  0.01006305  0.03871776]]
   
    '''
    num_kk = len(alpha_list)
    length_list = []
    for alpha in alpha_list:
        length_list.append(len(alpha))
    max_dk = np.amax(length_list)
    
    bernoulli = np.zeros((num_kk, max_dk))
    for row, alpha in enumerate(alpha_list):
        bernoulli[row, :len(alpha)] = np.array(np.random.dirichlet(alpha,1))
    
    return bernoulli    

def get_permuted_array_clusters_from_freqdata(catdata):
    '''obtain a list of clusters from frequencies of
    clusterings
    INPUT:
    catdata = 
    array([ 1,  1,  3,  8,  5, 22])
    
    OUTPUT:
    An permuted list of integers,
    
     permuted_clusters = [5 3 5 3 5 5 4 4 3 5 5 5 5 5 3 5 4
     5 5 5 5 2 5 3 4 1 4 5 5 5 5 0 5 2 2 3 5
     3 3 5]
    
    '''
    clusterlist = []
    for label, freq in enumerate(catdata):
        nut = [label]*freq
        clusterlist.append(nut)
    chainie = itertools.chain(*clusterlist)    
    cha = list(chainie)
    ordered_clusters = np.array(cha)
    #ordered_clusters = [0 1 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 5 
    #5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5]
    permuted_clusters = np.random.permutation(ordered_clusters)
    return permuted_clusters

def obtain_superclusters_from_bernoulli(bernoulli_matrix, spikes_in_cluster):
    '''
    obtain frequencies from a bernoulli matrix
    
    USAGE:
    superbern = obtain_superclusters_from_bernoulli(bernie, 20)
    
    INPUT:
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    
    [[ 0.21997748  0.03165307  0.00891037  0.42548762  0.01305409  0.02507721  0.00815188  0.26768828]
     [ 0.12970013  0.20768556  0.02535184  0.01659094  0.07383631  0.05132797  0.0165413   0.47896595]
     [ 0.04593969  0.02844991  0.408686    0.17328529  0.07582795  0.16893794  0.09758434  0.00128888]
     [ 0.0249658   0.25122204  0.06272984  0.02720584  0.39969485  0.00766485  0.1795039   0.04701287]
     [ 0.31600424  0.02470532  0.36808944  0.09132546  0.00434213  0.00530767  0.01774205  0.17248369]
     [ 0.21842491  0.44896848  0.13658852  0.04496099  0.0625706   0.06548539  0.00638156  0.01661955]]
       
     spikes_in_cluster, number of spikes in the global cluster 
     represented by the bernoulli matrix
     
     OUTPUT:
     superclusters for this particular bernoulli matrix
     
     [[7 1 4 2 2 1]
      [3 7 2 4 2 0]
      [3 7 2 6 0 1]
      [7 1 4 4 2 1]
      [0 7 2 1 3 5]
      [0 1 2 7 0 2]
      [7 7 4 3 3 3]
      [1 7 6 2 2 1]
      [0 7 2 6 7 5]
      [1 7 2 1 0 1]
      [3 7 5 4 0 2]
      [3 5 3 4 2 1]
      [0 0 3 4 2 1]
      [7 1 2 0 2 1]
      [7 7 0 5 7 4]
      [3 2 3 0 1 2]
      [3 4 0 3 2 2]
      [3 4 2 2 3 1]
      [3 0 2 4 0 0]
      [0 0 2 6 0 1]]
     
    '''
    
    single_supercluster = np.zeros((spikes_in_cluster, bernoulli_matrix.shape[0] ),dtype = int)
    for rr in np.arange(bernoulli_matrix.shape[0]):
        freqdata = np.random.multinomial(spikes_in_cluster,bernoulli_matrix[rr]) #these are the frequencies
        nut = get_permuted_array_clusters_from_freqdata(freqdata)
        single_supercluster[:,rr]= get_permuted_array_clusters_from_freqdata(freqdata)
    
    return single_supercluster

def generate_single_supercluster_points(alpha, num_kkruns, num_spikes_in_cluster):
    '''
    INPUT: 
    
    alpha = np.array([1, 1, 2, 2, 1, 0.5, 1, 0.02, 50]) 
    9 different local cluster labels
    len(alpha) = number of local clusters for kkrun
    
    num_kkruns
    num_spikes_in_cluster
    
    USAGE:
    superbernie, bernie = tc.generate_single_supercluster_points(alpha, 12, 40) 
    12 kk_runs, 40 points
    
    OUTPUT:
    superbernie = 
     [[3 8 6 1 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 8 3 4 1 6 0 8]
      [3 8 6 0 7 8 3 4 0 6 0 8]
      [3 8 5 8 7 8 3 4 0 6 5 8]
      [3 8 6 6 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 6 3 2 0 3 0 2]
      [3 8 6 6 7 8 6 4 0 6 0 8]
      [3 8 6 8 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 6 3 5 0 8 0 8]
      [3 8 6 8 7 8 3 4 0 3 0 8]
      [7 8 6 8 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 8 3 4 0 6 0 6]
      [7 8 6 8 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 6 3 4 0 6 0 8]
      [3 8 6 6 7 8 3 4 0 6 0 8]
      [3 8 6 8 7 8 5 4 7 8 1 2]
      [3 8 6 8 7 8 3 4 0 6 5 8]
      [3 8 1 8 7 3 3 5 0 6 0 2]
      [4 8 6 8 7 8 3 5 0 6 0 8]
      [3 8 6 8 7 8 3 4 0 6 0 8]] 
    
    The rows of bernie, the matrix of Bernoilli parameters, sum to 1. 
    bernie.shape = (num_kkruns, len(alpha) = number of local clusters for kkrun)
    
    bernie = 
     [[  1.97476409e-02   6.26633412e-04   3.41944891e-02   8.75579502e-01
    3.09546300e-02   3.88577673e-13   1.77238879e-02   1.28347604e-02
    8.33845592e-03]
  [  1.75283350e-03   3.20282443e-03   1.91103866e-04   5.67202097e-02
      5.26570487e-07   2.25990742e-02   6.99545505e-03   3.02289855e-05
      9.08507744e-01]
  [  6.73896633e-03   1.36408605e-02   2.59523834e-09   2.61289461e-03
      1.98078034e-02   7.70776964e-03   9.03957631e-01   3.13960742e-02
      1.41379979e-02]
  [  3.51588384e-02   2.73391051e-02   6.61641937e-03   7.68960047e-61
      1.19332400e-02   2.99571673e-03   4.13394517e-02   2.02948477e-02
      8.54322381e-01]
  [  5.12590333e-03   1.72364993e-02   1.96557085e-17   3.42962286e-02
      2.18896783e-02   3.55532461e-03   1.11148256e-02   8.92853971e-01
      1.39275691e-02]
  [  1.38331538e-02   1.39529942e-04   1.45865674e-02   1.27343067e-02
      6.48988925e-30   5.54500120e-02   1.58410260e-01   1.50289203e-02
      7.29817250e-01]
  [  6.40143618e-03   1.59065882e-02   1.00992563e-02   9.33167101e-01
      4.24248326e-03   7.69377122e-03   1.57750832e-02   6.71428027e-03
      4.54035149e-97]
  [  1.07282324e-02   4.42613112e-03   6.10705471e-03   1.71261661e-02
      8.60851836e-01   3.21988152e-02   1.56657026e-02   5.16143632e-18
      5.28960619e-02]
  [  9.11660345e-01   1.29985025e-03   4.88792794e-03   3.29784540e-11
      1.96914167e-03   1.98346516e-02   3.66883066e-06   5.95467865e-02
      7.97627941e-04]
  [  3.04940185e-03   1.65812058e-02   1.01433949e-02   8.70077350e-02
      4.83003059e-03   1.73721781e-33   8.51776545e-01   7.01127903e-04
      2.59105590e-02]
  [  8.67028551e-01   2.27347683e-02   2.98768505e-02   9.63805308e-03
      7.74280131e-03   5.35591612e-02   5.32261436e-03   4.09719994e-03
      8.92503570e-94]
  [  1.65386043e-02   9.42182217e-03   1.38970304e-02   1.15559482e-03
      3.94356615e-19   1.14964884e-02   1.80085938e-02   2.51493651e-03
      9.26966930e-01]]
    '''
    alpha_permie = generate_kkrun_samples_from_permuted_dirichlet(alpha, num_kkruns)
    aleph = list(alpha_permie)
    #print(aleph)
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    #print(bernie)
    #print(np.sum(bernie, axis = 1))
    superbern = obtain_superclusters_from_bernoulli(bernie, num_spikes_in_cluster)
    
    return superbern, bernie
        
def create_mixture_categorical_data(alpha_list, total_num_spikes, num_kkruns,  alpha_mix = None, mixture_weights = None):
    '''
    Create a mixture of categorical data
    INPUT:
    --------
    alpha_list - Bernoulli generators for each cluster
    total_num_spikes
    num_kkruns
    
    optional (need to specify one of either):
    alpha_mix - Bernoulli generator for mixture weights
    mixture_weights
    
    OUTPUT:
    ---------
    Mixture, a Python dictionary with keys:
    alpha_list - Bernoulli generators
    mixture weights
    mixture numbers,
    superclusters,
    bernoulli_matrices for each cluster
    '''
    
    if mixture_weights is not None:
        assert np.sum(mixture_weights) == 1
    elif alpha_mix is not None:    
        mixture_weights = np.array(np.random.dirichlet(alpha_mix,1))[0]
        
    mixture = {'mixture_weights':mixture_weights, 
                'alpha_list':alpha_list,
                 }
    
    mixture_numbers = np.random.multinomial(total_num_spikes,mixture_weights)
    assert np.sum(mixture_numbers) == total_num_spikes
    
    bernoulli_list = []
    supercluster_list = []
    for cluster, num_spikes_in_cluster in enumerate(mixture_numbers):
        superbernie, bernie = \
                    generate_single_supercluster_points(alpha_list[cluster], num_kkruns, num_spikes_in_cluster)
        bernoulli_list.append(bernie)
        supercluster_list.append(superbernie)
    superclusters = np.vstack(supercluster_list)    
       # counter += num_spikes_in_cluster
    
    mixture.update({'mixture_numbers': mixture_numbers})
    mixture.update({'superclusters':superclusters})
    mixture.update({'bernoulli_matrices':bernoulli_list})
    return mixture          
  
       
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