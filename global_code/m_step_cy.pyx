#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy

from cython cimport integral, floating

from libcpp.vector cimport vector

cpdef create_cluster_bern(floating[:,:] cluster_bern,
                          integral[:,:] supersparsekks,
                          integral[:] super_start,
                          integral[:] super_end,
                          integral[:]   spikes,
                          integral num_spikes_in_cluster):
    cdef integral pp, p, i, k, d, num_nontrivial
    
    for pp in range(num_spikes_in_cluster):
        p = spikes[pp]
        #dims = supersparsekks[super_start[p]:super_end[p]][:,0]  = array([0, 2, 3, 4, 5, 6])
        #spclust = supersparsekks[super_start[p]:super_end[p]][:,1]  = array([ 3,  1,  1, 56, 13,  1])
        num_nontrivial = super_end[p]-super_start[p] #dims where a KKrun took place
        for i in range(num_nontrivial):
            k = supersparsekks[super_start[p]+i,0]
            d = supersparsekks[super_start[p]+i,1]
            cluster_bern[k,d] += 1
            cluster_bern[k,0] += -1
                       

