import pickle
import numpy as np
from supercluster import *
from IPython import embed

'''
An illustrative example of sparsification:

NOTE: Sparsification will FAIL if there are any vectors that are all zero. [0,0,0,0,0,0,0] is not permitted in 
this example

little = np.array([[1,4,5,6,0,0,0], [0,0,0,7,7,1,1], [0,0,0,7,7,1,1], [1,4,5,6,0,0,0], [2,0,1,1,1,1,3], [4,5,0,0,0,12,10],

                   [8,8,8,8,4,4,4], [8,8,8,8,4,4,4],[3,0,1,1,56,13,1],[3,0,1,1,56,13,1],

                  [3,0,0,0,0,0,1], [3,0,0,0,0,0,1],[0,0,0,7,7,1,1],[3,0,1,1,56,13,1], [1,4,5,7,0,0,0],[1,4,5,8,0,0,0]], dtype = int)
This is a numpy array of superclusters

superlittle = sparsify_superclusters(little)
superlittle.offsets = array([ 0,  4,  8, 12, 16, 22, 26, 33, 40, 46, 52, 54, 56, 60, 66, 70, 74])

cute_little = superlittle.to_sparse_data()

cute_little.supersparsekks.shape = (41, 2)
cute_little.supersparsekks = array(
      [[ 0,  1],
       [ 1,  4],
       [ 2,  5],
       [ 3,  6],
       [ 0,  1],
       [ 1,  4],
       [ 2,  5],
       [ 3,  7],
       [ 0,  1],
       [ 1,  4],
       [ 2,  5],
       [ 3,  8],
       [ 0,  2],
       [ 2,  1],
       [ 3,  1],
       [ 4,  1],
       [ 5,  1],
       [ 6,  3],
       [ 0,  3],
       [ 2,  1],
       [ 3,  1],
       [ 4, 56],
       [ 5, 13],
       [ 6,  1],
       [ 0,  3],
       [ 6,  1],
       [ 0,  4],
       [ 1,  5],
       [ 5, 12],
       [ 6, 10],
       [ 0,  8],
       [ 1,  8],
       [ 2,  8],
       [ 3,  8],
       [ 4,  4],
       [ 5,  4],
       [ 6,  4],
       [ 3,  7],
       [ 4,  7],
       [ 5,  1],
       [ 6,  1]])

cute_little.super_start = array([ 0, 37, 37,  0, 12, 26, 30, 30, 18, 18, 24, 24, 37, 18,  4,  8])
cute_little.super_end = array([ 4, 41, 41,  4, 18, 30, 37, 37, 24, 24, 26, 26, 41, 24,  8, 12])

cute_little.num_spikes = 16

cute_little.D_k = array([ 8,  8,  8,  8, 56, 13, 10])
from vispy.plot import Fig
>>> fig = Fig()
>>> ax = fig[0, 0]  # this creates a PlotWidget
>>> ax.plot([[0, 1], [0, 1]])


'''


def sparsify_superclusters(superclusters):
    num_spikes = superclusters.shape[0]
    num_KKs = superclusters.shape[1] #120 in this case

    #Run this loop first to determine how large total_used_KKs
    #needs to be
    total_used_KKs = 0
    for i in np.arange(num_spikes):
        inds, = (superclusters[i,:]>0).nonzero() 
        #the comma after inds turns the array into a list
        total_used_KKs += len(inds)

    sparse_all_KKs = np.zeros(total_used_KKs, dtype=int)
    sparse_all_indices = np.zeros(total_used_KKs, dtype=int)
    offsets = np.zeros(num_spikes+1, dtype=int)
    curoff = 0

    for i in np.arange(num_spikes):
        inds, = (superclusters[i,:]>0).nonzero() 
        sparse_all_KKs[curoff:curoff+len(inds)] = superclusters[i,inds]
        sparse_all_indices[curoff:curoff+len(inds)] = inds
        offsets[i] = curoff
        curoff += len(inds)    
    offsets[-1] = curoff    #final value of offsets
    
    D_k = np.amax(superclusters, axis = 0) 
    #Number of clusters in the kth KK run
    #this assumes that the cluster labels are monotonic,
    # i.e. no gaps 0, 1, 2, ..., D(k)
    
    return GlobalSparseData(sparse_all_KKs, sparse_all_indices, offsets, num_KKs, D_k)

def reduce_supermasks_from_arrays(Ostart, Oend, I, K):
    #Ostart = silly.offsets[:-1]
    #Oend = silly.offsets[1:]
    #I = silly.sparse_all_indices
    #K = silly.sparse_all_KKs
    
    x = np.arange(len(Ostart))
    # converting the array to a string allows for a lexicographic compare
    # the details of the comparison are irrelevant as long as it is
    # consistent (for sorting) and never equal if the underlying arrays
    # are unequal
    #tuparray = np.array([tup for tup in zip(nut.sparse_all_indices[8:16],nut.sparse_all_KKs[8:16] )])
    #x = np.array(sorted(x, key=lambda p: I[Ostart[p]:Oend[p]].tostring()), dtype=int)
    x = np.array(sorted(x, key=lambda p: np.array([tup for tup in zip(I[Ostart[p]:Oend[p]],K[Ostart[p]:Oend[p]])]).tostring()), dtype=int)
    y = np.empty_like(x)
    y[x] = np.arange(len(x)) # y is the inverse of x as a permutation
    # step 2: iterate through all indices and add to collection if the
    # indices have changed
    oldstr = None
    new_indices = []
    start = np.zeros(len(Ostart), dtype=int)
    end = np.zeros(len(Ostart), dtype=int)
    curstart = 0 #current start
    curend = 0 #current end
    for i, p in enumerate(x):
        #curind = I[Ostart[p]:Oend[p]]
        curind = np.array([tup for tup in zip(I[Ostart[p]:Oend[p]],K[Ostart[p]:Oend[p]])])
       # if i<5:
       #    print(curind)
       #     print(curind.shape)
        #    print(len(curind))
        curstr = curind.tostring()
        if curstr!=oldstr:
            #if curind == []:
                #embed()
            new_indices.append(curind)
            oldstr = curstr
            curstart = curend
            curend += len(curind)
        start[i] = curstart
        end[i] = curend
    # step 3: convert into start, end
    #print(new_indices)
    #embed()
    sparse_indices = np.concatenate(new_indices, axis = 0)
    unique_superclusters, frequency = np.unique(start, return_counts = True)
    unique_superclusters_ends = np.unique(end)                                           
    #num_unique_superclusters = len(new_indices)
    return sparse_indices, new_indices, start[y], end[y], unique_superclusters, unique_superclusters_ends,frequency, x, y
    #return sparse_indices, new_indices, start[y], end[y]
#, num_unique_superclusters  


def reduce_supermasks(superdata):
    # step 1: sort into lexicographical order of masks
    start = superdata.offsets[:-1]#set of starts
    end = superdata.offsets[1:]# set of ends
    I = superdata.sparse_all_indices
    K = superdata.sparse_all_KKs
    return reduce_supermasks_from_arrays(start, end, I, K)

def superclusters_with_over_nspikes(super_frequency, spikes_per_cluster):
    '''Find the indices of the superclusters which contain more than spikes_per_cluster
        spikes
        #print(inds)
    ##print(super_frequency[inds])
    #for a, b in zip(silly.super_start[inds], silly.super_end[inds]):
    #    print(silly.supersparsekks[a:b,:])
        '''
    inds, = (super_frequency > spikes_per_cluster).nonzero()
    return inds
  
def superclusters_with_exactly_nspikes(super_frequency, spikes_per_cluster):
    inds, = (super_frequency == spikes_per_cluster).nonzero()
    return inds  
  
def test_supercluster_mask_difference(supersparsekks, superstart, superend, candidate_id, candidate_end):
    d =supercluster_mask_difference_pyt(supersparsekks, superstart, superend, candidate_id, candidate_end)
    return d

class GlobalSparseData(object):
    '''Sparse data for global superclustering'''
    def __init__(self,
                sparse_all_KKs, sparse_all_indices, 
                 offsets, numKKs, D_k):
        self.sparse_all_KKs = sparse_all_KKs
        self.sparse_all_indices = sparse_all_indices
        self.offsets = offsets
        self.num_KKruns = numKKs
        self.D_k = D_k

    def to_sparse_data(self):
        values_start = self.offsets[:-1]
        values_end = self.offsets[1:]
        supersparsekks, superlistkks, super_start, super_end, unique_superclusters,\
              unique_superclusters_ends, super_frequency, x, y  =  reduce_supermasks(self)
        self.supersparsekks = supersparsekks
        self.superlistkks = superlistkks
        self.super_start = super_start
        self.super_end = super_end
        self.unique_superclusters = unique_superclusters
        self.unique_superclusters_ends = unique_superclusters_ends
        self.super_frequency = super_frequency
        self.ordering_perm = x
        self.inv_ordering_perm = y
        self.num_spikes = super_start.shape[0]
        return SparseData(supersparsekks, super_start, super_end, self.num_KKruns, self.D_k)
       # return supersparsekks, superlistkks, super_start, super_end,unique_superclusters, \
    # unique_superclusters_ends, super_frequency, x, y

    def supercluster_distribution(self):
        max_freq = np.amax(self.super_frequency)
        #print(max_freq)
        min_freq = np.amin(self.super_frequency)
        #print(min_freq)
        biggersupercluster_indict = {}
        distribution_superclusterdict = {}
        for spikes_per_cluster in np.arange(min_freq-1, max_freq):
            indie = superclusters_with_over_nspikes(self.super_frequency, spikes_per_cluster)
            biggersupercluster_indict[spikes_per_cluster] = indie
            indie_dist = superclusters_with_exactly_nspikes(self.super_frequency, spikes_per_cluster)
            distribution_superclusterdict[spikes_per_cluster] = indie_dist
        self.biggersupercluster_indict = biggersupercluster_indict 
        self.distribution_superclusterdict = distribution_superclusterdict
        return biggersupercluster_indict, distribution_superclusterdict

    def clump_fine_clustering(self, clusters_withatleast):
        clusters = np.full(self.num_spikes, -1, dtype = int) 
        #If we get any clusters labelled -1 then there is some horrific bug!
        allspikes = np.arange(self.num_spikes)
        try:
            chosen_superclusterids = self.biggersupercluster_indict[clusters_withatleast]
        except KeyError:     
            max_atleast = np.amax(list(self.biggersupercluster_indict.keys()))
            print('There are no clusters with more than %d points \n'%(clusters_withatleast))
            print('Setting clusters_withatleast to %d'%(max_atleast))
            chosen_superclusterids = self.biggersupercluster_indict[max_atleast]
            
        candidate_ids_start = self.unique_superclusters[chosen_superclusterids]
        candidate_ids_end = self.unique_superclusters_ends[chosen_superclusterids]
        cand_cluster_label = {startid:cluster_label for cluster_label, startid in enumerate(candidate_ids_start)}   
        #print(cand_cluster_label)
        clump_clustering(clusters, candidate_ids_start, candidate_ids_end, self.supersparsekks, self.super_start, self.super_end,
                        allspikes, self.num_KKruns, cand_cluster_label)
        return clusters, cand_cluster_label    
    
    def noise_dump_clustering(self, clusters_withatleast):
        clusters = np.full(self.num_spikes, -1, dtype = int) 
        #If we get any clusters labelled -1 then there is some horrific bug!
        allspikes = np.arange(self.num_spikes)
        try:
            chosen_superclusterids = self.biggersupercluster_indict[clusters_withatleast]
        except KeyError:     
            max_atleast = np.amax(list(self.biggersupercluster_indict.keys()))
            print('There are no clusters with more than %d points \n'%(clusters_withatleast))
            print('Setting clusters_withatleast to %d'%(max_atleast))
            chosen_superclusterids = self.biggersupercluster_indict[max_atleast]
            
        candidate_ids_start = self.unique_superclusters[chosen_superclusterids]
        candidate_ids_end = self.unique_superclusters_ends[chosen_superclusterids]
        cand_cluster_label = {startid:cluster_label for cluster_label, startid in enumerate(candidate_ids_start)}   
        #print(cand_cluster_label)
        noisedump_clustering(clusters, candidate_ids_start, candidate_ids_end, self.supersparsekks, self.super_start, self.super_end,
                        allspikes, self.num_KKruns, cand_cluster_label)
        return clusters, cand_cluster_label    

class SparseData(object):
    '''
    Notes:
    - Assumes that the spikes are in sorted mask order, 
    '''
    def __init__(self, supersparsekks, super_start, 
                   super_end, num_KKruns, D_k):
         
        # Data arrays
        self.supersparsekks = supersparsekks
        self.super_start = super_start
        self.super_end = super_end
        self.num_KKruns = num_KKruns
        self.D_k = D_k
        
        # Derived data
        self.num_spikes = len(self.super_start)
        
    def subset(self, spikes):
        return SparseData(self.supersparsekks, self.super_start[spikes],
                 self.super_end[spikes], self.num_KKruns, 
                   self.D_k,
                   )

    