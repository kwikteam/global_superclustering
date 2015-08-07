import numpy as np
import time
from e_step_cy import *
#find_sublogresponsibility

def compute_cluster_sublogresponsibility(kk, cluster, weights, cluster_bern):
    '''compute the numerator of the responsibilities
       bern[cluster, KKrun, localclust] '''
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_kkruns = kk.num_KKruns
    num_spikes = kk.num_spikes
    spikes = kk.get_spikes_in_cluster(cluster)
    num_spikes_in_cluster = len(spikes)
    
    filler = np.log(weights[cluster])- num_kkruns*np.log(num_spikes_in_cluster)
    clust_sublogresponsibility = np.full(num_spikes,filler)
    print(filler)
    start_time = time.time()
    for p in range(num_spikes):
        find_sublogresponsibility(clust_sublogresponsibility,cluster_bern,supersparsekks, super_start, super_end, num_spikes,num_kkruns)
        
        #nonzero_kkruns = supersparsekks[super_start[p]:super_end[p],0]
        #for k in range(num_kkruns):
            #if k not in nonzero_kkruns:
               ## print('k = ',k)
                #clust_sublogresponsibility[p] += np.log(cluster_bern[k,0])
        #num_nontrivial = super_end[p]-super_start[p]
        #for i in range(num_nontrivial):
            #kkrun = supersparsekks[super_start[p]+i,0]
            #dlocal = supersparsekks[super_start[p]+i,1]
            ##print('cluster_bern [%g,%g] = '%(kkrun, dlocal), cluster_bern[kkrun, dlocal])
            #clust_sublogresponsibility[p] += np.log(cluster_bern[kkrun, dlocal]) 
    time_taken = time.time()-start_time
    print('Time taken for computing clust_sublogresponsibility %.2f s' %(time_taken))
    
    print('clust_sublogresponsibility for cluster %g'%(cluster), clust_sublogresponsibility)
    return clust_sublogresponsibility        

#def compute_responsibilities(kk, cluster, weights, bern):