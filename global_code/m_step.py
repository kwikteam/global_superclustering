import numpy as np

def compute_cluster_bern(kk,cluster,max_Dk):
    data = kk.data
    supersparsekks = data.supersparsekks
    super_start = data.super_start
    super_end = data.super_end
    
    num_kkruns = kk.num_KKruns
    
    spikes = kk.get_spikes_in_cluster(cluster)
    num_spikes_in_cluster = len(spikes)
    #bern
    cluster_bern_norm = np.zeros((num_kkruns, max_Dk+1), dtype = np.float32 )
    cluster_bern = np.zeros((num_kkruns, max_Dk+1), dtype = np.float32 )
    cluster_bern[:,0] = num_spikes_in_cluster #Initialise by setting 
    #everything to the zero cluster
    
    for pp in range(num_spikes_in_cluster):
        p = spikes[pp]
        #dims = supersparsekks[super_start[p]:super_end[p]][:,0]  = array([0, 2, 3, 4, 5, 6])
        #spclust = supersparsekks[super_start[p]:super_end[p]][:,1]  = array([ 3,  1,  1, 56, 13,  1])
        num_nontrivial = super_end[p]-super_start[p] #dims where a KKrun took place
        for i in range(num_nontrivial):
            k = supersparsekks[super_start[p]+i,0]
            d = supersparsekks[super_start[p]+i,1]
            cluster_bern[k,d] = 1
            cluster_bern[k,0] += -1
    cluster_bern_norm /= num_spikes_in_cluster
    return cluster_bern, cluster_bern_norm 
            
    
     
    