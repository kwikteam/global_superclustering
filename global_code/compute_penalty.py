import numpy as np

__all__ = ['compute_penalties']

def compute_penalty(kk, clusters):
      if clusters is None:
          clusters = kk.clusters
      num_cluster_membs = np.array(np.bincount(clusters), dtype=int)
      alive = num_cluster_membs>0
      num_clusters = np.sum(alive)
      print('num_cluster_members', num_cluster_membs)
      print('num_clusters', num_clusters)
      #This now only depends on the number of clusters 
      #cluster_penalty = np.zeros(num_clusters)
      num_spikes = kk.num_spikes
      #D_k = kk.D_k
      num_kkruns = kk.num_KKruns
      num_bern_params = kk.num_bern_params
      num_bern_params_used = num_bern_params[np.nonzero(num_cluster_membs>0)]
      print('num_bern_params_used = ', num_bern_params_used)
      #num_bern_params =  [70 71 63 63 64 62 83 79] for 8 clusters
      penalty_k = kk.penalty_k
      penalty_k_log_n = kk.penalty_k_log_n
      
      #mean_params = (np.sum(D_k)-num_kkruns)*num_clusters - 1
      #effective_params = bernoulli params + mixture weight
      effective_params = np.sum(num_bern_params_used)-num_kkruns*num_clusters + (num_clusters -1)
      
      penalty = (2*penalty_k*effective_params + penalty_k_log_n*effective_params*np.log(num_spikes)/2)
      print('penalty = ', penalty)

      #do_compute_penalty(cluster_penalty, num_spikes, clusters,
      #                     penalty_k, penalty_k_log_n)
      #may not need to import
      return penalty
    