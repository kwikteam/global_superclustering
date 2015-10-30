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
    cdef integral p, i1, id0, id1
  #  cdef numpy.ndarray allkkrun_dims = numpy.arange(num_kkruns, dtype = numpy.int)
  #  cdef numpy.ndarray origin_superclusters = numpy.zeros(num_kkruns, dtype = numpy.int)

   # all_zero_sum = 0
    #for idx in range(num_kkruns):
    #    if numpy.isfinite(log_cluster_bern[idx,0]):
   #         all_zero_sum += log_cluster_bern[idx,0]
    for p in range(num_spikes):
        #clust_sublogresponsibility[p] += all_zero_sum
        i1 = super_start[p]
        #while i1<super_end[p]:
        for i1 in range(super_start[p], super_end[p]):    
            id0 = supersparsekks[i1,0]
            id1 = supersparsekks[i1,1]
            clust_sublogresponsibility[p] += log_cluster_bern[id0,id1]
            if numpy.isfinite(log_cluster_bern[id0,0]):
                clust_sublogresponsibility[p] -= log_cluster_bern[id0,0]
           # i1 += 1
            
cpdef find_all_sublogresponsibility(floating[:,:] prelogresponsibility,
                               floating[:,:,:] log_bern,
                               integral[:,:] supersparsekks,
                               integral[:] super_start,
                               integral[:] super_end,
                               integral num_spikes, 
                               integral num_kkruns,
                               integral num_clusters):
    cdef integral p, i1, id0, id1, cluster
  #  cdef numpy.ndarray allkkrun_dims = numpy.arange(num_kkruns, dtype = numpy.int)
  #  cdef numpy.ndarray origin_superclusters = numpy.zeros(num_kkruns, dtype = numpy.int)

   # all_zero_sum = 0
    #for idx in range(num_kkruns):
    #    if numpy.isfinite(log_cluster_bern[idx,0]):
   #         all_zero_sum += log_cluster_bern[idx,0]
    for p in range(num_spikes):
        #clust_sublogresponsibility[p] += all_zero_sum
        
        i1 = super_start[p]
        #while i1<super_end[p]:
        for i1 in range(super_start[p], super_end[p]):    
            id0 = supersparsekks[i1,0]
            id1 = supersparsekks[i1,1]
            for cluster in range(num_clusters):
                prelogresponsibility[cluster,p] += log_bern[cluster,id0,id1]
                if log_bern[cluster,id0,0]>-numpy.inf:
                #if numpy.isfinite(log_bern[cluster,id0,0]):
                    prelogresponsibility[cluster,p] -= log_bern[cluster,id0,0]
            #i1 += 1        

#cpdef top_two(ndarray):
#   cdef integral i 
#    for i in range(ndarray.shape[0]):
        

    
cpdef do_p_loop_log_p_and_assign(floating[:,:] prelogresponsibility,
                              floating[:] log_p,
                              floating[:] log_p_best,
                              floating[:] log_p_second_best,
                              integral[:] clusters,
                              integral[:] clusters_second_best,
                              integral num_spikes,
                              integral num_clusters,
                              char only_evaluate_current_clusters,
                              ):
    cdef integral pp, p, sortbest, secondsortbest
    cdef floating cur_log_p_best, cur_log_p_second_best
    for pp in range(num_spikes):
        p = pp
        sortbest = 0
        secondsortbest = 1
        #smallest = -prelogresponsibility[0,p]
        #secondsmallest = -prelogresponsibility[1,p]
        if -prelogresponsibility[1,p] < -prelogresponsibility[0,p]:
            sortbest = 1
            secondsortbest = 0
        for i in range(2,num_clusters):
            if -prelogresponsibility[i,p]< -prelogresponsibility[secondsortbest,p]:
                if -prelogresponsibility[i,p]< -prelogresponsibility[sortbest,p]:
                    secondsortbest = sortbest   
                    sortbest = i  
                else:
                    secondsortbest = i
  
        #print(sortbest, secondsortbest)
        #print(prelogresponsibility[sortbest,p],prelogresponsibility[secondsortbest,p])
        
        #orderfrombest = np.argsort(-prelogresponsibility[:,p])
        
        log_p[p] = prelogresponsibility[sortbest,p] #prelogresponsibility[orderfrombest[0],p]
        cur_log_p_best = log_p_best[p]
        if not only_evaluate_current_clusters:
            cur_log_p_second_best = log_p_second_best[p]
        if not only_evaluate_current_clusters:        
            if cur_log_p_best > log_p[p]:
                #print('WARNING: cluster assignment for point p not changing')
                #embed()
                #kk.log_p_best[p] does not change
                if cur_log_p_second_best > log_p[p]:
                    log_p_second_best[p] = cur_log_p_second_best
                else:    
                    log_p_second_best[p] = log_p[p]
                #cluster assignment for point p does not change    
            else:    
                log_p_best[p] = log_p[p] 
            #    if not (len(orderfrombest) <2) and (prelogresponsibility[secondsortbest,p] > -numpy.inf):#np.isfinite(prelogresponsibility[orderfrombest[1],p]):
                if not (sortbest != secondsortbest) and (prelogresponsibility[secondsortbest,p] > -numpy.inf):#np.isfinite(prelogresponsibility[orderfrombest[1],p]):                    
                    log_p_second_best[p] = prelogresponsibility[secondsortbest,p] #prelogresponsibility[orderfrombest[1],p]
                else: 
                    log_p_second_best[p] = prelogresponsibility[sortbest,p] #prelogresponsibility[orderfrombest[0],p]
                clusters[p] = sortbest #orderfrombest[0]
                #clusters reassigned due to improvement 
                #print('clusters being reassigned') 
                #if not (len(orderfrombest) <2):
                if sortbest != secondsortbest:
                    clusters_second_best[p] = secondsortbest #orderfrombest[1]
        else:
            log_p_best[p] = log_p[p]           
                               
                               
