from phy.io import KwikModel
from phy.cluster.session import Session
import phy
phy.__version__
import parallel_global as pg
from klustakwik2 import *
import numpy as np
import pickle
import sys
import os
import copy
import time
from IPython.parallel import Client
from IPython import embed

scriptname = os.path.basename(__file__)
print('Running script: ', os.path.basename(__file__))

#os.system('ipcluster start -n 8')
#sys.exit()
basefolder = '/mnt/zserver/Data/multichanspikes/M140528_NS1/20141202/'
littlename = '20141202_all'  
basename =  basefolder + littlename   
kwik_path = basename + '.kwik'

model = KwikModel(kwik_path)
#session = Session(kwik_path)

#Make an old-fashioned .fet and .fmask file
numb_spikes_to_use = 3700
if numb_spikes_to_use ==None:
    masky = model.masks[:]
    fetty = model.features[:]
else:    
    masky = model.masks[:numb_spikes_to_use+1]
    fetty = model.features[:numb_spikes_to_use+1]

triplemasky = np.repeat(masky,3, axis = 1)
print(masky.shape)
print(fetty.shape)
numspikes = masky.shape[0]

#derived_basename = basename
outputpath = '/home/skadir/globalphy/nicktest/'
derived_basename = outputpath + 'nick_global_%g'%(numspikes)

fmaskbase = derived_basename + '.fmask.1'
fetbase = derived_basename + '.fet.1'
pg.write_fet(fetty, fetbase)
pg.write_mask(triplemasky,fmaskbase)

#sys.exit()

#Make a full adjacency graph
print('Making full adjacency graph')
full_adjacency  = pg.make_full_adjacency(model.probe.adjacency)
print(full_adjacency)

#active_channels = model.channels

#Channel order dictionary
channel_order_dict = pg.make_channel_order_dict(model.channel_order)
#embed()
#For each channel find the indices of the points which are unmasked on this channel 
print('Making globalcl_dict')   
globalcl_dict = {}
globalcl_dict = pg.find_unmasked_points_for_channel(masky,channel_order_dict,full_adjacency,globalcl_dict)


#compute indices of spike groups by taking the union of the unmasked indices
#for each channel belonging to the spike group  
globalcl_dict = pg.find_unmasked_spikegroup(full_adjacency,globalcl_dict)

#Make dictionary of subset features and masks 
fetmask_dict = {}
pg.make_subset_fetmask(fetmask_dict, fetty, triplemasky, channel_order_dict, full_adjacency, globalcl_dict)    

#Run MKK on each subset with the default value of the parameters
script_params = default_parameters.copy()
script_params.update(
drop_last_n_features=0,
save_clu_every=None,
run_monitoring_server=False,
save_all_clu=False,
debug=True,
start_from_clu=None,
use_noise_cluster=False,
use_mua_cluster=False,
subset_schedule=None,
)

print('Running KK on subsets')
print(script_params)
print(script_params.keys())

# Start timing

shank = 1

drop_last_n_features = script_params.pop('drop_last_n_features')
save_clu_every = script_params.pop('save_clu_every')
run_monitoring_server = script_params.pop('run_monitoring_server')
save_all_clu = script_params.pop('save_all_clu')
debug = script_params.pop('debug')
num_starting_clusters = script_params.pop('num_starting_clusters')
start_from_clu = script_params.pop('start_from_clu')
use_noise_cluster = script_params.pop('use_noise_cluster')
use_mua_cluster = script_params.pop('use_mua_cluster')
subset_schedule = script_params.pop('subset_schedule')

start_time = time.time()
raw_data = load_fet_fmask_to_raw(derived_basename, shank, drop_last_n_features=drop_last_n_features)
log_message('debug', 'Loading data from .fet and .fmask file took %.2f s' % (time.time()-start_time))
data = raw_data.to_sparse_data()
log_message('info', 'Number of spikes in data set: '+str(data.num_spikes))
log_message('info', 'Number of unique masks in data set: '+str(data.num_masks))
kk = KK(data, use_noise_cluster=use_noise_cluster, use_mua_cluster=use_mua_cluster, **script_params)

supercluster_info = {}
supercluster_info.update({'kk_sub':{}, 'sub_spikes':{}})
for channel in full_adjacency.keys():
    kk_sub, spikes = kk.subset_features(list(fetmask_dict['pcs_inds'][channel]))
    supercluster_info['kk_sub'].update({channel:kk_sub})
    supercluster_info['sub_spikes'].update({channel:spikes})

pickle_info = [supercluster_info, full_adjacency, channel_order_dict]
with open('%s_supercluster_info.p'%(derived_basename),'wb') as gg:
    pickle.dump(pickle_info,gg)

#sys.exit()
#Run KK2 on all the subsets
numKK = len(full_adjacency.keys()) #equal to the number of channels
superclusters = np.zeros((fetty.shape[0],numKK))
c = Client(profile = 'default')
lbv = c.load_balanced_view()
lbv.block = True
#with c[:].sync_imports():
#    import os, sys
#    sys.path.append('/home/skadir/globalphy/global_superclustering/global_code/')
    #import parallel_global
#    from parallel_global import run_subset_KK, squared
#    from klustakwik2 import clustering

#c[:].execute('import klustakwik2')
c[:].execute('import os, sys')
c[:].execute('sys.path.append(\'/home/skadir/globalphy/global_superclustering/global_code/\')')
c[:].execute('from klustakwik2 import clustering')
c[:].execute('from parallel_global import run_subset_KK')
#c[:].execute('from parallel_global import squared')
#c[:].execute('print(parallel_global.__file__)')

#c[:]['supercluster_info[\'kk_sub\']']  =   supercluster_info['kk_sub']
c[:]['supercluster_info']  =   supercluster_info
c[:]['full_adjacency'] = full_adjacency
#embed()
#c[:].run(print('supercluster_info'))
#v = c[:]
#v.map()
print('About to parallelize')
#embed()
start_time2 = time.time()
#supercluster_results = lbv.map(lambda channel: supercluster_info['kk_sub'][channel].cluster_mask_starts(),full_adjacency.keys())
#supercluster_results = lbv.map(lambda channel: run_subset_KK(channel),full_adjacency.keys())
#supercluster_results = lbv.map(lambda channel: squared(channel),full_adjacency.keys())

#def parallel_clustering(chan):
#    clustering = run_subset_KK(supercluster_info['kk_sub'][channel])
#    return clustering

supercluster_results = lbv.map(lambda channel: run_subset_KK(supercluster_info['kk_sub'][channel]),full_adjacency.keys())
#supercluster_results = lbv.map(parallel_clustering, full_adjacency.keys()), 
time_taken_parallel = time.time()-start_time2
print('Time taken for parallel clustering %.2f s' %(time.time()-start_time2))
#for channel in full_adjacency.keys():
#    supercluster_info['kk_sub'][channel].cluster_mask_starts()
#    supercluster_info['kk_sub'][channel].cluster_mask_starts()
#embed()
   
for i, channel in enumerate(full_adjacency.keys()):
    superclusters[supercluster_info['sub_spikes'][channel],i] = supercluster_results[i]+1
    #We add 1 because we don't want 0 to be an acceptable cluster label
    


# for channel in full_adjacency.keys():
#     scriptname = basename+'%g'%(channel)
#     scriptstring = '''import klustakwik2 as *
#               '''
#     scriptfile = open('%s.sh' %(scriptname),'w')
#     scriptfile.write(scriptstring)
#     scriptfile.close()
#     changeperms='chmod 777 %s.sh' %(scriptname)
#     os.system(changeperms)   

superinfo = [time_taken_parallel, full_adjacency, channel_order_dict,globalcl_dict,supercluster_info,supercluster_results, superclusters]
#print(supercluster_results)
#superinfo = [supercluster_results]
with open('%s_supercluster.p'%(derived_basename), 'wb') as g:
    pickle.dump(superinfo, g)    