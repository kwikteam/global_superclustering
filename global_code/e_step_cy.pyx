#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy

from cython cimport integral, floating

from libcpp.vector cimport vector

cpdef sum_finite(nparray):
    finite_sum = numpy.sum(nparray[numpy.isfinite(nparray)])
    return finite_sum


cpdef find_sublogresponsibility(floating[:] clust_sublogresponsibility,
                               floating[:,:] log_cluster_bern,
                               integral[:,:] supersparsekks,
                               integral[:] super_start,
                               integral[:] super_end,
                               integral num_spikes, 
                               integral num_kkruns ):
    cdef integral p, i1
  #  cdef numpy.ndarray allkkrun_dims = numpy.arange(num_kkruns, dtype = numpy.int)
  #  cdef numpy.ndarray origin_superclusters = numpy.zeros(num_kkruns, dtype = numpy.int)

   # all_zero_sum = 0
    #for idx in range(num_kkruns):
    #    if numpy.isfinite(log_cluster_bern[idx,0]):
   #         all_zero_sum += log_cluster_bern[idx,0]
    for p in range(num_spikes):
        #clust_sublogresponsibility[p] += all_zero_sum
        i1 = super_start[p]
        while i1<super_end[p]:
            id0 = supersparsekks[i1,0]
            id1 = supersparsekks[i1,1]
            clust_sublogresponsibility[p] += log_cluster_bern[id0,id1]
            if numpy.isfinite(log_cluster_bern[id0,0]):
                clust_sublogresponsibility[p] -= log_cluster_bern[id0,0]
            i1 += 1
            
        
    
        
    
         
    
    
                               
                               
