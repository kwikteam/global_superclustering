import numpy as np

def compute_cluster_subresponsibility(kk, cluster, weights, cluster_bern):
    '''compute the numerator of the responsibilities
       bern[cluster, KKrun, localclust] '''
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_spikes = kk.num_spikes
    
    clust_subresponsibility = np.zeros(num_spikes)
    for p in range(num_spikes):
        clust_subresponsibility[p] = weights[cluster]
        num_nontrivial = super_end[p]-super_start[p]
        for i in range(num_nontrivial):
	    kkrun = supersparsekks[super_start[p]+1,0]
	    dlocal = supersparsekks[super_start[p]+1,1]
            clust_subresponsibility[p] *= cluster_bern[kkrun, dlocal]
            
    return clust_subresponsibility        

#def compute_responsibilities(kk, cluster, weights, bern):