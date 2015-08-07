import pickle
import numpy as np
from supercluster import *

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
            new_indices.append(curind)
            oldstr = curstr
            curstart = curend
            curend += len(curind)
        start[i] = curstart
        end[i] = curend
    # step 3: convert into start, end
    #print(new_indices)
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
        return supersparsekks, superlistkks, super_start, super_end,unique_superclusters, \
     unique_superclusters_ends, super_frequency, x, y

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

       