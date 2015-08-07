import numpy as np

def compute_cluster_sublogresponsibility(kk, cluster, weights, cluster_bern):
    '''compute the numerator of the responsibilities
       bern[cluster, KKrun, localclust] '''
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_spikes = kk.num_spikes
    spikes = kk.get_spikes_in_cluster(cluster)
    num_spikes_in_cluster = len(spikes)
    
    clust_sublogresponsibility = np.zeros(num_spikes)
    for p in range(num_spikes):
        clust_sublogresponsibility[p] = np.log(weights[cluster])
        num_nontrivial = super_end[p]-super_start[p]
        for i in range(num_nontrivial):
            kkrun = supersparsekks[super_start[p]+1,0]
            dlocal = supersparsekks[super_start[p]+1,1]
            clust_sublogresponsibility[p] += np.log(cluster_bern[kkrun, dlocal]) - np.log(num_spikes_in_cluster)
            
    return clust_sublogresponsibility        

#def compute_responsibilities(kk, cluster, weights, bern):