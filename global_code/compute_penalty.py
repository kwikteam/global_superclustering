import numpy as np

__all__ = ['compute_penalties']

def compute_penalty(kk, clusters):
      if clusters is None:
          clusters = kk.clusters
      num_clusters = np.amax(clusters)+1
      #This now only depends on the number of clusters 
      #cluster_penalty = np.zeros(num_clusters)
      
      D_k = kk.D_k
      num_kkruns = kk.num_KKruns
      
      penalty_k = kk.penalty_k
      penalty_k_log_n = kk.penalty_k_log_n
      num_spikes = np.zeros(num_clusters, dtype=int)
      mean_params = (np.sum(D_k)-num_kkruns)*num_clusters - 1
      
      penalty = (penalty_k*mean_params*2 + penalty_k_log_n*mean_params*log(total_num_spikes)/2)

      #do_compute_penalty(cluster_penalty, num_spikes, clusters,
      #                     penalty_k, penalty_k_log_n)
      #may not need to import
      return penalty
    