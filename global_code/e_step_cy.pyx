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
    cdef numpy.ndarrray allkkrun_dims = numpy.arange(num_kkruns, dtype = numpy.int)
    cdef numpy.ndarrray origin_superclusters = numpy.zeros(num_kkruns, dtype = numpy.int)
    for p in range(num_spikes):        
        origin_superclusters[supersparsekks[super_start[p]:super_end[p],0]] = supersparsekks[super_start[p]:super_end[p],1]
        #clust_sublogresponsibility[p] += numpy.sum(log_cluster_bern[:,origin_superclusters]) 
        clust_sublogresponsibility[p] += numpy.sum(log_cluster_bern[allkkrun_dims,origin_superclusters]) 
        origin_superclusters[supersparsekks[super_start[p]:super_end[p],0]] = 0               
                               
                               
                               
