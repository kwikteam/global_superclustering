import numpy as np
import time
from IPython import embed
from six.moves import range
from e_step_cy import *
find_sublogresponsibility

def sum_finite(nparray):
    finite_sum = np.sum(nparray[np.isfinite(nparray)])
    return finite_sum

def compute_subresponsibility(kk, weights, log_bern,num_clusters):
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_kkruns = kk.num_KKruns
    num_spikes = kk.num_spikes
    num_cluster_members = kk.num_cluster_members
    
    prelogresponsibility = np.zeros((num_clusters, num_spikes), dtype = float)
    
    for cluster in range(num_clusters):
        spikes = kk.get_spikes_in_cluster(cluster)
        num_spikes_in_cluster = len(spikes) 
        all_zero_sum = sum_finite(log_bern[cluster,:,0])
        filler = np.log(weights[cluster])- num_kkruns*np.log(num_spikes_in_cluster) + all_zero_sum
        prelogresponsibility[cluster,:] = np.full(num_spikes,filler, dtype = np.float64)
    #print(clust_sublogresponsibility.shape)
    #print(filler)
    #embed()
    start_time = time.time()
    find_all_sublogresponsibility(prelogresponsibility,log_bern,supersparsekks, super_start, super_end, num_spikes,num_kkruns, num_clusters)
    time_taken = time.time()-start_time
    print('Time taken for computing full prelogresponsibility %.2f s' %(time_taken))   
    return prelogresponsibility
    
def compute_cluster_subresponsibility(kk, cluster, weights, log_cluster_bern):
    '''compute the numerator of the responsibilities
       log_cluster_bern.shape = (num_ '''
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_kkruns = kk.num_KKruns
    num_spikes = kk.num_spikes
    num_cluster_members = kk.num_cluster_members
    spikes = kk.get_spikes_in_cluster(cluster)
    num_spikes_in_cluster = len(spikes) #equal to num_cluster_members[cluster]
    
    #prodexcludeself = np.prod(num_cluster_members)/num_spikes_in_cluster
    
    #clust_subresponsibility = np.full(num_spikes,weights[cluster])
    all_zero_sum = sum_finite(log_cluster_bern[:,0])
    filler = np.log(weights[cluster])- num_kkruns*np.log(num_spikes_in_cluster) + all_zero_sum
    clust_sublogresponsibility = np.full(num_spikes,filler, dtype = np.float64)
    print(clust_sublogresponsibility.shape)
    #print(filler)
    #embed()
    start_time = time.time()
    
    
    
    #all_zero_sum = np.sum(log_cluster_bern[:,0])
    #total_diff = 0
    #max_diff = 0
    
    #all_zero_sum = sum_finite(log_cluster_bern[:,0])
    #for p in range(num_spikes):
        #clust_sublogresponsibility[p] += all_zero_sum
        #clust_sublogresponsibility[p] += np.sum(log_cluster_bern[supersparsekks[super_start[p]:super_end[p],0],supersparsekks[super_start[p]:super_end[p],1]])
        #clust_sublogresponsibility[p] -= sum_finite(log_cluster_bern[supersparsekks[super_start[p]:super_end[p],0],0])
        
        #clust_sublogresponsibility[p] -= np.sum(log_cluster_bern[supersparsekks[super_start[p]:super_end[p],0],0])
       # total_diff += super_end[p]-super_start[p]
       # if super_end[p]-super_start[p]> max_diff:
       #     max_diff = super_end[p]-super_start[p]
       #     print(p)
    
    ##BEST PYTHON CODE so far---------------------------
    #origin_superclusters = np.zeros(num_kkruns, dtype = int)
    #allkkrun_dims = np.arange(num_kkruns, dtype = int)
    
    #for p in range(num_spikes):        
        ##origin_superclusters = np.zeros(num_kkruns, dtype = int)
        #origin_superclusters[supersparsekks[super_start[p]:super_end[p],0]] = supersparsekks[super_start[p]:super_end[p],1]
        #clust_sublogresponsibility[p] += np.sum(log_cluster_bern[allkkrun_dims,origin_superclusters])
        #origin_superclusters[supersparsekks[super_start[p]:super_end[p],0]] = 0
    ##----------------------------------------------    
        
        #embed()
        #nonzero_kkruns = supersparsekks[super_start[p]:super_end[p],0]
        #prezero_kkruns = np.arange(num_kkruns)
        #zero_kkruns = np.delete(prezero_kkruns, nonzero_kkruns)
        #clust_sublogresponsibility[p] += np.sum(log_cluster_bern[zero_kkruns,0])   
        #clust_sublogresponsibility[p] += np.sum(log_cluster_bern[supersparsekks[super_start[p]:super_end[p],0],supersparsekks[super_start[p]:super_end[p],1]])
        #embed()
        #clust_subresponsibility[p] *= np.prod(cluster_bern[zero_kkruns,0])   
        #clust_subresponsibility[p] *= np.prod(cluster_bern[supersparsekks[super_start[p]:super_end[p],0],supersparsekks[super_start[p]:super_end[p],1]])
    find_sublogresponsibility(clust_sublogresponsibility,log_cluster_bern,supersparsekks, super_start, super_end, num_spikes,num_kkruns)
        ##for k in np.arange(num_kkruns):
        #   # if k not in nonzero_kkruns:
        ##for k in zero_kkruns:    
        ##    clust_sublogresponsibility[p] += log_cluster_bern[k,0]
        ##num_nontrivial = super_end[p]-super_start[p]
        ##for i in np.arange(num_nontrivial):
            ##kkrun = supersparsekks[super_start[p]+i,0]
           # #dlocal = supersparsekks[super_start[p]+i,1]
           # #print(kkrun)
           # #print('cluster_bern [%g,%g] = '%(kkrun, dlocal), cluster_bern[kkrun, dlocal])
           # #clust_sublogresponsibility[p] += log_cluster_bern[kkrun, dlocal]
         
    time_taken = time.time()-start_time
    print('Time taken for computing clust_sublogresponsibility %.2f s' %(time_taken))   
    #print('clust_sublogresponsibility for cluster %g'%(cluster), clust_sublogresponsibility)
    #mean_start_end = total_diff/num_spikes
    #print('max_diff =' , max_diff)
    #print('mean_start_end = ', mean_start_end)
     
    return clust_sublogresponsibility

def compute_log_p_and_assign(kk, prelogresponsibility, 
                             only_evaluate_current_clusters):
    num_clusters = len(kk.num_cluster_members)
    num_kkruns = kk.num_KKruns
    num_spikes = kk.num_spikes

    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end

    log_p_best = kk.log_p_best
    log_p_second_best = kk.log_p_second_best
    
    clusters = kk.clusters
    clusters_second_best = kk.clusters_second_best
    old_clusters = kk.old_clusters
    log_p = np.zeros(num_spikes)
    
    #if only_evaluate_current_clusters:
    #    embed()
    #    candidates = kk.candidates
    
    #for p in np.arange(num_spikes):
    for pp in np.arange(num_spikes):
        p = pp
        #if not only_evaluate_current_clusters:
        #    p = pp
        #else:
        #    embed()
        #    p = candidates[pp]        
        #Fix bug where log_p_second_best is -inf
        orderfrombest = np.argsort(-prelogresponsibility[:,p])
        #print('prelogresponsibility[:,p],shape = ',prelogresponsibility[:,p].shape)
        #e.g. for a clustering with 15 clusters
        # prelogresponsibility[:,p]=

        #array([-17.0767553 , -16.62460313, -17.39244399, -21.39556163,
        #-17.95783375, -17.15554351, -18.9431675 , -17.22197569,
        #-17.98608194, -17.59084827, -18.33352782, -18.33929569,
        #-19.93217079, -17.73675297, -16.3812674 ])

        #orderfrombest = array([14,  1,  0,  5,  7,  2,  9, 13,  4,  8, 10, 11,  6, 12,  3])

    
        #embed()
 #       if len(orderfrombest) <2:
            #print('orderfrombest = ',orderfrombest)
            #print(-prelogresponsibility[:,p])
            #embed()
        #embed()    
        #kk.log_p_best[p] = prelogresponsibility[orderfrombest[0],p]
        log_p[p] = prelogresponsibility[orderfrombest[0],p]
        #log_p[p] = -16.381267402829636
        cur_log_p_best = log_p_best[p]
        if not only_evaluate_current_clusters:
            cur_log_p_second_best = log_p_second_best[p]
        
        #Fix bug where log_p_second_best is -inf
        # In this case, set log_p_second_best = log_p_best
        #print('only_evaluate_current_clusters = ', only_evaluate_current_clusters)
        #print(p)
        
        #embed()
        
        if not only_evaluate_current_clusters:        
            if cur_log_p_best > log_p[p]:
                print('WARNING: cluster assignment for point p not changing')
                embed()
                #kk.log_p_best[p] does not change
                if cur_log_p_second_best > log_p[p]:
                    kk.log_p_second_best[p] = cur_log_p_second_best
                else:    
                    kk.log_p_second_best[p] = log_p[p]
                #cluster assignment for point p does not change    
            else:    
                kk.log_p_best[p] = log_p[p] 
                if not (len(orderfrombest) <2) and np.isfinite(prelogresponsibility[orderfrombest[1],p]):
                    kk.log_p_second_best[p] = prelogresponsibility[orderfrombest[1],p]
                else: 
                    kk.log_p_second_best[p] = prelogresponsibility[orderfrombest[0],p]
                kk.clusters[p] = orderfrombest[0]
                #clusters reassigned due to improvement 
                #print('clusters being reassigned') 
                if not (len(orderfrombest) <2):
                    kk.clusters_second_best[p] = orderfrombest[1]
                
                #Alternative
                #kk.log_p_best[p] = log_p[p] 
                #if not (len(orderfrombest) <2) and np.isfinite(prelogresponsibility[orderfrombest[1],p]):
                    #kk.log_p_second_best[p] = prelogresponsibility[orderfrombest[1],p]
                #else: 
                    #kk.log_p_second_best[p] = prelogresponsibility[orderfrombest[0],p]
                #kk.clusters[p] = orderfrombest[0]
                ##clusters reassigned due to improvement 
                ##print('clusters being reassigned') 
                #if not (len(orderfrombest) <2):
                    #kk.clusters_second_best[p] = orderfrombest[1]                 
        else:
            kk.log_p_best[p] = log_p[p] 
            #return
            #return
            #embed()
            #if cur_log_p_best < log_p[p]:   
            #    print('OMG e_step reassignment')
                #kk.log_p_best[p] = log_p[p] 
                #kk.clusters[p] = orderfrombest[0]
                #clusters reassigned due to improvement - but this should not occur!
                
   # if only_evaluate_current_clusters:
    #    return
      
    #for p in np.arange(num_spikes):
    #    log_p = cluster_log_p
        
    #    cur_log_p_best = log_p_best[p]
    #    cur_log_p_second_best = 
        
        
        
    #full_step = kk.full_step
   
    #cluster_log_p = numpy.zeros(num_spikes)
    #candidates = zeros(0, dtype=int)
    
    #do_log_p_assign_computations(
                                  
                                  #log_p_best, log_p_second_best,
                                  #clusters, clusters_second_best, old_clusters,
                                  
                                  #num_, num_spikes, log_addition, cluster,
                                  #chol.block, chol.diagonal, chol.masked, chol.unmasked,
                                  #n_cpu,
                                  #cluster_log_p,
                                  #candidates,
                                  #only_evaluate_current_clusters,
                                  #)

    #if only_evaluate_current_clusters:
        #kk.log_p_best[candidates] = cluster_log_p[candidates]
    #elif kk.collect_candidates and hasattr(kk, 'old_log_p_best'):
        #max_quick_step_candidates = min(kk.max_quick_step_candidates,
            #kk.max_quick_step_candidates_fraction*kk.num_spikes*kk.num_clusters_alive)
        #candidates, = (cluster_log_p-kk.old_log_p_best<=kk.dist_thresh).nonzero()
        #kk.quick_step_candidates[cluster] = array(candidates, dtype=int)
        #num_candidates = sum(len(v) for v in itervalues(kk.quick_step_candidates))
        #if num_candidates>max_quick_step_candidates:
            #kk.collect_candidates = False
            #kk.quick_step_candidates.clear()
            #kk.force_next_step_full = True
            #if num_candidates>kk.max_quick_step_candidates:
                #kk.log('info', 'Ran out of storage space for quick step, try increasing '
                               #'max_quick_step_candidates if this happens often.')
            #else:
                #kk.log('debug', 'Exceeded quick step point fraction, next step '
                                #'will be full')

    #return kk.num_spikes-num_spikes
##def compute_responsibilities(kk, cluster, weights, bern):