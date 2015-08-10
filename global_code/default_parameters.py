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
     max_possible_clusters=1000,
     dist_thresh=log(10000.0),
     consider_cluster_deletion=True,
     break_fraction = 0.0,
     num_cpus=None,
     )
