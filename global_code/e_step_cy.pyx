#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy

from cython cimport integral, floating

from libcpp.vector cimport vector

cpdef find_sublogresponsibility(floating[:,:] clust_sublogresponsibility,
                               integral[:,:] log_cluster_bern,
                               integral[:,:] supersparsekks,
                               integral[:] super_start,
                               integral[:] super_end,
                               integral num_spikes, 
                               integral num_kkruns ):
    cdef integral p, k, i, d, kkrun, dlocal                               
    for p in range(num_spikes):
        nonzero_kkruns = supersparsekks[super_start[p]:super_end[p],0]
        for k in range(num_kkruns):
            if k not in nonzero_kkruns:
                clust_sublogresponsibility[p] += log_cluster_bern[k,0]
        num_nontrivial = super_end[p]-super_start[p]
        for i in range(num_nontrivial):
            kkrun = supersparsekks[super_start[p]+i,0]
            dlocal = supersparsekks[super_start[p]+i,1]
            clust_sublogresponsibility[p] += log_cluster_bern[kkrun, dlocal]                                
                               
                               
                               
