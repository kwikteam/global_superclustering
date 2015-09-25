'''
Default clustering parameters
'''

from numpy import log

default_parameters = dict(
     penalty_k=0.0,
     penalty_k_log_n=1.0,
     max_iterations=1000,
     num_starting_clusters=500,
     num_changed_threshold=0.05,
     split_first=20,
     split_every=40,
     fast_split=False,
     max_possible_clusters=1000,
     dist_thresh=log(10000.0),
     consider_cluster_deletion=True,
     break_fraction = 0.0,
     num_cpus=None,
     max_candidates = 100000000,  
     max_candidates_fraction = 0.4,
     max_split_iterations = None,
     min_points_split_cluster = 10,
     save_clu_every = None,
     run_monitoring_server = False,
     debug = False,
     save_all_clu = False,
     embed = False,
     use_cluster_penalty = False,
     save_prelogresponsibility = False,
     )
