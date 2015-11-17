#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy
from numpy.random import randint

from cython cimport integral, floating

from libcpp.vector cimport vector

cdef integral supercluster_mask_difference(integral[:,:] supersparsekks, integral start1, integral end1, integral start2, integral end2):
    cdef integral i1, i2, u1, u2, v1, v2, d
    #cdef integral n1 = end1-start1
    #cdef integral n2 = end2-start2
    i1 = start1
    i2 = start2
    d = 0
    while i1<end1 or i2<end2:
        if i1>=end1:
            # there now cannot be a match to u2, so we increase distance by 1
            d += 1
            i2 += 1
            continue
        if i2>=end2:
            # there now cannot be a match to u1, so we increase distance by 1
            d += 1
            i1 += 1
            continue
        u1 = supersparsekks[i1,0]  
        u2 = supersparsekks[i2,0]
        v1 = supersparsekks[i1,1]
        v2 = supersparsekks[i2,1]
        #u1 = unmasked[i1]
        #u2 = unmasked[i2]
        if (u1==u2 and v1==v2):
            # we match u1 and u2 so increment both pointers to put them out of the pool and since they're matched
            # we don't increase the distance
            i1 += 1
            i2 += 1
        elif (u1==u2 and v1!=v2):
            i1 += 1
            i2 += 1
            d += 1
        elif u1<u2:
            # we remove u1 from the pool and it didn't match, so we increase the distance
            d += 1
            i1 += 1
        elif u2<u1:
            # we remove u2 from the pool and it didn't match, so we increase the distance
            d += 1
            i2 += 1
    return d
  
cpdef integral supercluster_mask_difference_pyt(integral[:,:] supersparsekks, integral start1, integral end1, integral start2, integral end2):
    cdef integral d
    d = supercluster_mask_difference(supersparsekks, start1, end1, start2, end2)
    return d

cpdef clump_clustering(integral[:] clusters,
		     integral[:] candidate_ids_start,
		     integral[:] candidate_ids_end,#self.biggersupercluster_indict[155]
                     integral[:,:] supersparsekks,
                     integral[:] superstart,
                     integral[:] superend,
                     integral[:] allspikes,
                     integral numKKs,
                     dict cand_cluster_label,
                     ):
    cdef vector[integral] best_ids
    cdef vector[integral] candidate_ids
    cdef vector[integral] candidate_ends
    cdef integral p, best_distance, candidate_id, candidate_end, c_idx, d
    #found = dict()
    #end = dict()
    for p in allspikes:
        #supermask_startid = superstart[p]
        if superstart[p] in candidate_ids_start:
            # This spike belongs to one of the big superclusters already
            clusters[p] = cand_cluster_label[superstart[p]]
        else: # we have to find the closest supercluster (this is the computationally intensive bit!)
            best_distance = numKKs+1
            best_ids.clear()
            for c_idx in range(len(candidate_ids_start)):
                candidate_id = candidate_ids_start[c_idx]
                candidate_end = candidate_ids_end[c_idx]
                d = supercluster_mask_difference(supersparsekks, superstart[p], superend[p], candidate_id, candidate_end)  
                if d==best_distance:
                    best_ids.push_back(candidate_id)
                elif d<best_distance:
                    best_distance = d
                    best_ids.clear()
                    best_ids.push_back(candidate_id)
                best_id = best_ids[randint(best_ids.size())]   #if there are many equidistant superclusters, take a random one
                #print(best_id)
                clusters[p] = cand_cluster_label[best_id]
  #  return clusters 
  
cpdef noisedump_clustering(integral[:] clusters,
             integral[:] candidate_ids_start,
             integral[:] candidate_ids_end,#self.biggersupercluster_indict[155]
                     integral[:,:] supersparsekks,
                     integral[:] superstart,
                     integral[:] superend,
                     integral[:] allspikes,
                     integral numKKs,
                     dict cand_cluster_label,
                     ):
    
    cdef integral p, candidate_id, candidate_end, noisedump_cluster_id
    #found = dict()
    #end = dict()
    noisedump_cluster_id = len(candidate_ids_start)
    for p in allspikes:
        #supermask_startid = superstart[p]
        if superstart[p] in candidate_ids_start:
            # This spike belongs to one of the big superclusters already
            clusters[p] = cand_cluster_label[superstart[p]]
        else: # we just dump it in an extra cluster
                clusters[p] = noisedump_cluster_id
  #  return clusters      
  