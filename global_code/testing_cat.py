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
  
def generate_kkrun_samples_from_permuted_dirichlet(alpha, num_kkruns):
    '''
    INPUT:
    alpha = np.array([1, 1, 1, 5, 4, 9])
    num_kkruns = 6
    
    OUTPUT:
    alpha_permie_list = generate_kkrun_samples_from_permuted_dirichlet(alpha, 6)
         
    [array([ 1.,  1.,  9.,  5.,  4.,  1.]),
     array([ 1.,  9.,  1.,  4.,  5.,  1.]),
     array([ 1.,  1.,  1.,  5.,  4.,  9.]),
     array([ 1.,  5.,  1.,  9.,  1.,  4.]),
     array([ 1.,  5.,  1.,  4.,  9.,  1.]),
     array([ 1.,  1.,  5.,  9.,  4.,  1.])] 
    
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
    
    [array([ 1.,  1.,  9.,  5.,  4.,  1.]),
     array([ 1.,  9.,  1.,  4.,  5.,  1.]),
     array([ 1.,  1.,  1.,  5.,  4.,  9.]),
     array([ 1.,  5.,  1.,  9.,  1.,  4.]),
     array([ 1.,  5.,  1.,  4.,  9.,  1.]),
     array([ 1.,  1.,  5.,  9.,  4.,  1.])]
    
    OUTPUT:
    bernoulli matrix for a global cluster
    shape = (num_kkruns, num_categories_local clustering)
    
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    
    [[ 0.02266499  0.01479669  0.30521821  0.2752903   0.30059345  0.08143635]
     [ 0.00099761  0.44261836  0.05572671  0.19684414  0.14442124  0.15939194]
     [ 0.00929971  0.03771584  0.02180445  0.17430253  0.24106294  0.51581452]
     [ 0.19364511  0.19970504  0.04124662  0.44627955  0.0369358   0.08218787]
     [ 0.01455205  0.3583447   0.06311833  0.14590676  0.37465121  0.04342694]
     [ 0.02466864  0.10984797  0.10666448  0.51909835  0.16907111  0.07064944]]

    
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
    An ordered list of integers,
    ordered_clusters = [0 1 2 2 2 3 3 3 3 3 3 3 3 4 4 4 4 4 5 
    5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5]
    
    '''
    clusterlist = []
    for label, freq in enumerate(catdata):
        nut = [label]*freq
        clusterlist.append(nut)
    chainie = itertools.chain(*clusterlist)    
    cha = list(chainie)
    ordered_clusters = np.array(cha)
    return ordered_clusters

def obtain_superclusters_from_bernoulli(bernoulli_matrix, spikes_in_cluster):
    '''
    obtain frequencies from a bernoulli matrix
    
    USAGE:
    superbern = obtain_superclusters_from_bernoulli(bernie, 20)
    
    INPUT:
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    
    [[ 0.02266499  0.01479669  0.30521821  0.2752903   0.30059345  0.08143635]
     [ 0.00099761  0.44261836  0.05572671  0.19684414  0.14442124  0.15939194]
     [ 0.00929971  0.03771584  0.02180445  0.17430253  0.24106294  0.51581452]
     [ 0.19364511  0.19970504  0.04124662  0.44627955  0.0369358   0.08218787]
     [ 0.01455205  0.3583447   0.06311833  0.14590676  0.37465121  0.04342694]
     [ 0.02466864  0.10984797  0.10666448  0.51909835  0.16907111  0.07064944]]
     
     spikes_in_cluster, number of spikes in the global cluster 
     represented by the bernoulli matrix
     
     OUTPUT:
     superclusters for this particular bernoulli matrix
     
     array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1],
       [0, 2, 0, 0, 0, 3],
       [0, 2, 0, 2, 0, 3],
       [0, 4, 0, 2, 0, 4],
       [1, 4, 0, 2, 0, 5],
       [1, 4, 2, 2, 2, 5],
       [5, 4, 2, 2, 2, 5],
       [5, 4, 2, 3, 2, 5],
       [5, 4, 2, 5, 2, 5],
       [5, 4, 2, 5, 2, 5],
       [5, 4, 2, 5, 2, 5],
       [5, 5, 3, 5, 2, 5],
       [5, 5, 5, 5, 2, 5],
       [5, 5, 5, 5, 2, 5],
       [5, 5, 5, 5, 4, 5],
       [5, 5, 5, 5, 4, 5],
       [5, 5, 5, 5, 4, 5],
       [5, 5, 5, 5, 4, 5]])
     
    '''
    
    single_supercluster = np.zeros((spikes_in_cluster, bernoulli_matrix.shape[0] ),dtype = int)
    for rr in np.arange(bernoulli_matrix.shape[0]):
        freqdata = np.random.multinomial(spikes_in_cluster,bernoulli_matrix[rr]) #these are the frequencies
        nut = get_permuted_array_clusters_from_freqdata(freqdata)
        single_supercluster[:,rr]= get_permuted_array_clusters_from_freqdata(freqdata)
    
    return single_supercluster

def generate_single_supercluster_points(alpha,num_kkruns, num_spikes_in_cluster):
    alpha_permie = generate_kkrun_samples_from_permuted_dirichlet(alpha, num_kkruns)
    aleph = list(alpha_permie)
    print(aleph)
    bernie = obtain_bernoulli_from_dirichlet_alphalist(aleph)
    print(bernie)
    print(np.sum(bernie, axis = 1))
    superbern = obtain_superclusters_from_bernoulli(bernie, num_spikes_in_cluster)
    
    return superbern
        
          
  
       
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