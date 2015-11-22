from phy.io import KwikModel
#from phy.cluster.session import Session
from phy.session import Session
import phy
phy.__version__
from klustakwik2 import *
#import hashutils
import numpy as np
import pickle
#import tables as tb
import sys
import os
import copy
import time
from IPython.parallel import Client
from IPython import embed
import h5py
from phy.io.h5 import open_h5
from phy.io.kwik.creator import KwikCreator, _write_by_chunk, create_kwik

def open_h5(filename, mode=None):
    """Open an HDF5 file and return a File instance."""
    file = File(filename, mode=mode)
    file.open()
    return file

#def reduce_kwik_file_bad(kwik_path,filename, num_spikes):
    '''This function does not work'''
#    kwikfile = kwik_path + filename + '.kwik'
#    with tb.openFile(kwikfile, mode = 'a') as kwikfile:
#        samptimes = kwikfile.root.channel_groups._f_getChild('1').spikes.time_samples[:100]
#        kwikfile.root.channel_groups._f_getChild('1').spikes.time_samples= kwikfile.root.channel_groups._f_getChild('1').spikes.time_samples[:num_spikes]
#        kwikfile.root.channel_groups._f_getChild('1').spikes.time_fractional= kwikfile.root.channel_groups._f_getChild('1').spikes.time_fractional[:num_spikes]
        
        #kwikfile.root.channel_groups[0].spikes.time_samples = kwikfile.channel_groups[0].spikes.time_samples[:num_spikes]
        #kwikfile.channel_groups[0].spikes.time_fractional = kwikfile.channel_groups[0].spikes.time_fractional[:num_spikes]
       # kwikfile.

def reduce_kwik_file(kwik_path_dir, parent_filename, baby_filename, num_spikes):
    '''Create a smaller baby kwikfile with only a subset of the spikes
       but with all the same metadata'''
    basename =    os.path.join(kwik_path_dir, parent_filename)
    model = KwikModel(basename)
    babybasename = os.path.join(kwik_path_dir, baby_filename)
    kwdname = kwik_path_dir + parent_filename + '.raw.kwd'
    creator = KwikCreator(babybasename)
    creator.create_empty()
    creator.set_metadata('/application_data/spikedetekt',**model.metadata)
    
    #Derive properties from parent to be assigned to baby
    n_channels = model.n_channels
    n_features = model.n_features_per_channel
    babygroup = model.channel_group 
    spike_samples = model.spike_samples[:num_spikes]
    features = model.features[:num_spikes]
    masks = model.masks[:num_spikes]
    creator.add_spikes(group=babygroup,
                       spike_samples=spike_samples,
                       features=features.astype(np.float32),
                       masks=masks.astype(np.float32),
                       n_channels=n_channels,
                       n_features=n_features,
                       )
    #Add probe info
    probedict = model.probe._probe
    creator.set_probe(probedict)
    #Add kwd info
    sample_rate = model.sample_rate
    if os.path.isfile(kwdname):
        creator._add_recordings_from_kwd(kwdname,sample_rate=sample_rate,)
    else:
        print('associating recording from dat file')
        datname = kwik_path_dir + parent_filename  + '.dat'
        creator._add_recordings_from_dat([datname], sample_rate = sample_rate, n_channels = n_channels, dtype = np.int16)
        
def add_clustering_to_kwik(kwik_path_dir, filename, clustering_name, clu_array):
    basename =    os.path.join(kwik_path_dir, filename)
    model = KwikModel(basename)
    model.add_clustering(clustering_name, clu_array)

def retrieve_res_file_from_kwik(kwik_path_dir, filename, filepath):
    basename =    os.path.join(kwik_path_dir, filename)
    model = KwikModel(basename)
    samps = model.spike_samples
    write_res(samps, filepath)

def write_mask(mask, filename, fmt="%f"):
    fd = open(filename, 'w')
    fd.write(str(mask.shape[1])+'\n') # number of features
    fd.close()
    fd = open(filename, 'ab')
    np.savetxt(fd, mask, fmt=fmt)
    fd.close()

def write_fet(feats, filepath, fmt = "%f"):
    feat_file = open(filepath, 'w')
    #feats = np.array(feats, dtype=np.int32)
    #header line: number of features
    feat_file.write('%i\n' % feats.shape[1])
    feat_file.close()
    feat_file = open(filepath, 'ab')
    #next lines: one feature vector per line
    #np.savetxt(feat_file, feats, fmt="%i")
    np.savetxt(feat_file, feats, fmt=fmt)
    feat_file.close()  

def write_clu(clus, filepath, fmt = "%i"):
    '''write a clu file
    input: clus is a 1d or 2D numpy array of integers
    output: 
        top line: number of clusters (max cluster)
        mext lines: one integer per line
    '''
    clu_file = open(filepath, 'w')
    n_clu = clus.max()+1
    clu_file.write('%i\n'%n_clu)
    clu_file.close()
    clu_file = open(filepath, 'ab')
    #one cluster per line
    np.savetxt(clu_file, np.int16(clus), fmt = fmt)
    clu_file.close()

def load_clu(fname):
    return np.loadtxt(fname, skiprows=1, dtype=int)-1

def load_res(fname):
    return np.loadtxt(fname, dtype=int)
    
def write_res(samples, filepath, fmt = "%i"):
    '''input: 1D vector of times, shape = (n_times,) or (n_times, 1)
    output: writes .res file, which has integer sample numbers
    '''    
    res_file = open(filepath, 'ab')
    np.savetxt(res_file, samples, fmt = fmt)
    res_file.close()

def make_KK2script(KKparams, filebase, shanknum,  scriptname):
    
    #keylist = KKparams['keylist']
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    #KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    #KKlocation = KKparams['KKlocation']
    #scriptstring = KKlocation + ' '+ filebase + ' %g '%(str(shanknum))
    KKlocation = 'kk2_legacy'
    scriptstring = KKlocation + ' '+ filebase + ' %g '%(shanknum)
    for KKey in KKparams.keys(): 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' '+ KKey +'='+ str(KKparams[KKey])
    
    print(scriptstring)
    scriptfile = open('%s.sh' %(scriptname),'w')
    scriptfile.write(scriptstring)
    scriptfile.close()
    changeperms='chmod 777 %s.sh' %(scriptname)
    os.system(changeperms)
    
    return scriptstring    

def make_KKscript(KKparams, filebase, scriptname):
    
    keylist = KKparams['keylist']
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    #KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    KKlocation = KKparams['KKlocation']
    shanknum = KKparams['shanknum']
    scriptstring = KKlocation + ' '+ filebase + ' %g '%(shanknum)
    for KKey in keylist: 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' -'+ KKey +' '+ str(KKparams[KKey])
    
    print(scriptstring)
    scriptfile = open('%s.sh' %(scriptname),'w')
    scriptfile.write(scriptstring)
    scriptfile.close()
    changeperms='chmod 777 %s.sh' %(scriptname)
    os.system(changeperms)
    
    return scriptstring

def make_KKscript_supercomp(KKparams, filebase, scriptname, supercomparams, localoutputpath, sendonlyscript = False):
    '''Create bash script on Legion required to run KlustaKwik
    supercomparams = {'time':'36:00:00','mem': '2G', 'tmpfs':'10G'}
    
    '''

    keylist = KKparams['keylist']
    
    #keylist = ['MaskStarts','MaxPossibleClusters','FullStepEvery','MaxIter','RandomSeed',
    #           'Debug','SplitFirst','SplitEvery','PenaltyK','PenaltyKLogN','Subset',
    #           'PriorPoint','SaveSorted','SaveCovarianceMeans','UseMaskedInitialConditions',
     #          'AssignToFirstClosestMask','UseDistributional']

    #KKlocation = '/martinottihome/skadir/GIT_masters/klustakwik/MaskedKlustaKwik'  
    scriptfilebase = 'super_'+scriptname+'_comp' #Legion no longer accepting names that start with a digit!
    supercompstuff = '''#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=%s
#$ -l mem=%s
#$ -l tmpfs=%s
#$ -N %s
#$ -wd /home/smgxsk1/Scratch/
cd $TMPDIR
'''%(supercomparams['time'],supercomparams['mem'],supercomparams['tmpfs'],scriptfilebase)
    
    KKsupercomplocation = supercompstuff +  '/home/smgxsk1/MKK_versions/klustakwik/MaskedKlustaKwik'
    scriptstring = KKsupercomplocation + ' /home/smgxsk1/Scratch/global/'+ filebase + ' %g '%(KKparams['shanknum'])
    for KKey in keylist: 
        #print '-'+KKey +' '+ str(KKparams[KKey])
        scriptstring = scriptstring + ' -'+ KKey +' '+ str(KKparams[KKey])
    
    print(scriptstring)
    scriptfile = open('%s/%s.sh' %(localoutputpath, scriptfilebase),'w')
    scriptfile.write(scriptstring)
    scriptfile.close()
    #outputdir = ' /chandelierhome/skadir/hybrid_analysis/mariano/'
    #changeperms='chmod 777 %s.sh' %(scriptname)
    if sendonlyscript:
        sendout = 'scp '+ localoutputpath + scriptfilebase + '.sh '+ 'smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/global/'
    else:    
        sendout = 'scp '+ localoutputpath + scriptfilebase + '.sh ' +\
    localoutputpath +scriptname + '.fet.1 ' + localoutputpath + scriptname + '.fmask.1 '+ 'smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/global/'
    print(sendout)
    os.system(sendout)
    
    return scriptstring

def run_on_supercomp_ind(outlist, kkgroupname, localoutputpath):
    #if kwik ==False:
    #    if outlist ==None:
     #       outlist = rkk.one_param_varyKK_ind(hybdatadict, SDparams, defaultKKparams, paramtochange, listparamvalues) 
    #else:
    #    if outlist ==None:
     #       outlist = rkk.one_param_varyKK_ind_kwik(hybdatadict, SDparams, defaultKKparams, paramtochange, listparamvalues,kwik)            
 
    outputdir = '/home/smgxsk1/Scratch/global/'
    qsubscript = '''import os
os.system(\''''
    for k, basefilename in enumerate(outlist):
        #qsubscriptn = qsubscript +  basefilename + '.fet.1'
        #qsubscriptn = qsubscript +  basefilename + '.fmask.1'
        #qsubscript = qsubscript + 'ln -s %s.fet.%g %s.fet.%g; ' %(outlist[0][0][:-33],defaultKKparams['shanknum'],basefilename,defaultKKparams['shanknum'])
        #qsubscript = qsubscript + 'ln -s %s.fmask.%g %s.fmask.%g; ' %(outlist[0][0][:-33],defaultKKparams['shanknum'],basefilename,defaultKKparams['shanknum'])
        qsubscript = qsubscript + 'qsub '+ outputdir + 'super_' + basefilename + '_comp.sh; ' 
        
    
    qsubscript = qsubscript + '\')'
    qsubscriptname = kkgroupname
    qsubfilename = localoutputpath+'%s_submission.py' %(qsubscriptname)
    scriptfile = open(qsubfilename,'w')
    scriptfile.write(qsubscript)
    scriptfile.close()
    print('Creating qsub script: ', qsubfilename)
    print('Sending it to Legion ')
    print(qsubscript)
    sendout = 'scp '+ qsubfilename+ ' smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/global/'
    os.system(sendout)
    
    

    
def retrieve_from_supercomp(outlist, kkgroupname, localoutputpath):
    
    outputdir = '/home/smgxsk1/Scratch/global/'
    bringback = '''import os
os.system(\''''
    for k, basefilename in enumerate(outlist):
        bringback = bringback + 'scp '+ outputdir + basefilename + '.clu.1 ' + outputdir + basefilename + '.klg.1 ' 
        
    bringback = bringback + 'skadir@chandelier.cortexlab.net:' + localoutputpath + '\')'
    retrievalscriptname = kkgroupname
    retrievefilename = localoutputpath+'%s_retrieve.py' %(retrievalscriptname)
    scriptfile = open(retrievefilename,'w')
    scriptfile.write(bringback)
    scriptfile.close()
    print('Creating retrieval script: ', retrievefilename)
    print('Sending it to Legion ')
    print(bringback)
    sendout = 'scp '+ retrievefilename + ' smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/global/'
    os.system(sendout)  

#def reduce_space_on_supercomp_ind(hybdatadict, SDparams,defaultKKparams, paramtochange, listparamvalues, extralabel = None):

    #outlistKK = rkk.one_param_varyKK_ind(hybdatadict, SDparams, defaultKKparams, paramtochange, listparamvalues)
    #outputdir = '/home/smgxsk1/Scratch/'
    #bringback = '''import os
#os.system(\''''
    #for k, basefilename in enumerate(outlistKK[0]):
        #bringback = bringback + 'scp '+ outputdir + basefilename + '.clu.1 ' + outputdir + basefilename + '.klg.1 ' 
        
    #bringback = bringback + 'skadir@chandelier.cortexlab.net:/home/skadir/testklusta/mariano/\')'
    #retrievalscriptname = outlistKK[0][0][:-33]
    #retrievefilename = hybdatadict['output_path']+'%s_%s_retrieve.py' %(retrievalscriptname,extralabel)
    #scriptfile = open(retrievefilename,'w')
    #scriptfile.write(bringback)
    #scriptfile.close()
    #print 'Creating retrieval script: ', retrievefilename
    #print 'Sending it to Legion '
    #print bringback
    #sendout = 'scp '+ retrievefilename+ ' smgxsk1@legion.rc.ucl.ac.uk:/home/smgxsk1/Scratch/'
    #os.system(sendout)  

    
def pca_licate_indices(channel_list, num_pcs):
    ordered_chanlist = np.sort(channel_list)
    modlist = [(num_pcs*ordered_chanlist + i) for i in np.arange(num_pcs)]
    pc_chans = np.sort(np.concatenate(modlist))
    #print(pc_chans)
    #np.sort(np.concatenate([3*np.sort(list(model.probe.adjacency[6]))+0,
    #3*np.sort(list(model.probe.adjacency[6]))+1,
    #3*np.sort(list(model.probe.adjacency[6]))+2]))
    #nut = pca_licate_indices(list(model.probe.adjacency[6]),3)
    #print(nut)
    return pc_chans
        
def make_full_adjacency(adj):
    '''Make a full adjacency graph by adding 
       reference channel to its neighbours
        adj = model.probe.adjacency
        return full_adjacency'''
    full_adjacency = copy.deepcopy(adj)
    for channel in full_adjacency.keys():
        full_adjacency[channel].add(channel)    
    return full_adjacency
    
def make_channel_order_dict(chorder): 
    '''
    chorder is model.channel_order
    In [13]: model.channel_order
    Out[13]: 
    array([  7,  39,   8,  41,   6,  38,   9,  42,   5,  37,  10,  43,   4,
            36,  11,  44,  35,  12,  45,   2,  34,  13,  46,   1,  33,  14,
            47,   0,  32,  15,  31,  63,  16,  49,  30,  62,  17,  50,  29,
            61,  18,  51,  28,  60,  19,  52,  59,  27,  53,  21,  58,  26,
            54,  22,  57,  25,  55,  23,  56,  24,  71, 103,  72, 105,  70,
           102,  73, 106,  74, 101,  69, 107,  75, 100,  68, 108,  99,  67,
           109,  77,  98,  66, 110,  78,  97,  65, 111,  79,  96,  64,  80,
           127,  95, 113,  81, 126,  94, 114,  82, 125,  93, 115,  83, 124,
            92, 116, 123,  91, 117,  85, 122,  90, 118,  86, 121,  89, 119,
            87, 120,  88], dtype=int32)

    In [14]: channel_order_dict[38]
    Out[14]: 5
    
    '''
    channel_order_dict = {}
    for j in np.arange(len(chorder)):
        #channel_order_dict[j] = chorder[j]  
        channel_order_dict[chorder[j]] = j
    return channel_order_dict     
  
def make_inverse_dict(dictionaer):
    '''Only works for a bijective dictionary'''
    inv_dict = {v:k for k, v in dictionaer.items()}

def give_value(dictionaer,k):
    v = dictionaer[k]
    return v
    
def find_unmasked_points_for_channel(masks,channel_order_dict,fulladj,globalcl_dict, threshold=None): 
    '''For each channel find the indices of the points
     which are unmasked on this channel 
     fulladj = full_adjacency
     masks - array of masks'''
    
    if threshold == None:
        threshold =0.001
    globalcl_dict.update({'unmasked_indices':{}})
    for channel in fulladj.keys():
        unmasked = np.where(masks[:,channel_order_dict[channel]]>= threshold)
        globalcl_dict['unmasked_indices'].update({channel:unmasked})
    return globalcl_dict    
        #print(model.probe.adjacency[channel])    

def find_unmasked_spikegroup(fulladj,globalcl_dict):  
    '''compute indices of spike groups by taking the union of the unmasked indices
    for each channel belonging to the spike group '''      
    globalcl_dict.update({'unmasked_spikegroup':{}})
    for channel in fulladj.keys():
        unmaskedunion = []
        for chan in list(fulladj[channel]): # e.g. model.probe.adjacency[5] = {1,2,4,5}
            unmaskedunion = np.union1d(unmaskedunion,globalcl_dict['unmasked_indices'][chan][0])
        unmaskedunion = np.array(unmaskedunion, dtype = np.int32)    
        globalcl_dict['unmasked_spikegroup'].update({channel:unmaskedunion})  
    return globalcl_dict

def filter_spike_groups(masks,channel_order_dict,fulladj,globalcl_dict, filter_thresh):
    '''
    computer the intersection of the fmask of every spike  assigned to a spike group by
    the function find_unmasked_spikegroup() above, with the set of channels
    composing the group. Eg. for filter_thresh =2:
    
      |  X   X   X   X |                  S_i  Intersection   Participation
    ------------------------------------|-----|-------------|---------------     
    1 |  1   1   0   0 |  0   0   0   0 |  3  |     2       |     Y
    0 |  0   1   1   1 |  1   1   0   0 |  5  |     3       |     Y
    0 |  0   1   0   0 |  0   0   0   0 |  1  |     1       |     Y
    0 |  0   0   0 0.5 |  1   1   1   1 | 4.5 |     1       |     N
    
    (note masks needn't be integers)
    '''
    
    for channelgp in fulladj.keys():
        spikegroup_chans = fulladj[channelgp]
        filtered_unmasked = []
        print('num spikes in group %g = ' %(channelgp), len(globalcl_dict['unmasked_spikegroup'][channelgp]))
        for spike_ind in globalcl_dict['unmasked_spikegroup'][channelgp]:
            masky = masks[spike_ind]
            ordchans = []
            for chan in spikegroup_chans:
                ordchans.append(channel_order_dict[chan])
            intersect_no = np.sum(masky[ordchans])
            total_mask_length = np.sum(masks[spike_ind])
            if (intersect_no>=filter_thresh):
                filtered_unmasked.append(spike_ind)
            elif (total_mask_length <=filter_thresh):
                filtered_unmasked.append(spike_ind)
            elif np.any(masky[ordchans] ==1):
                filtered_unmasked.append(spike_ind) #fix bug where spikes were excluded altogether from all kkruns
        globalcl_dict['unmasked_spikegroup'].update({channelgp:filtered_unmasked})
        print('num spikes in group %g after filtering = ' %(channelgp), len(globalcl_dict['unmasked_spikegroup'][channelgp]))
    return globalcl_dict    
               
def make_subset_fetmask(fetmask_dict, fetty, triplemasky, channel_order_dict,fulladj, globalcl_dict, writefetmask = False, basename = None):
    '''No longer necessary with the new subset feature in KK2,
    but kept here in case. Make .fet and .fmask files for the
    subsets'''  
    #fetmask_dict = {}
    fetmask_dict.update({'group_fet':{}, 'group_fmask':{}, 'pcs_inds':{}})
    for channel in fulladj.keys():
        ordchans = [channel_order_dict[x] for x in list(fulladj[channel])]
        pcs_inds = pca_licate_indices(ordchans,3)
        #pcs_inds = pca_licate_indices(list(fulladj[channel]),3)
        fetmask_dict['pcs_inds'].update({channel:pcs_inds})
        fetty_group = fetty[:,pcs_inds]
        fetty_little = fetty_group[globalcl_dict['unmasked_spikegroup'][channel],:] 
        fetmask_dict['group_fet'].update({channel:fetty_little})
        masky_group = triplemasky[:,pcs_inds]
        masky_little = masky_group[globalcl_dict['unmasked_spikegroup'][channel],:] 
        fetmask_dict['group_fmask'].update({channel:masky_little})
        
        if writefetmask == True:
            fetty_little_name = basename + '.fet.'+ str(channel) 
            masky_little_name = basename + '.fmask.'+ str(channel)
            print('writing file ',fetty_little_name)
            write_fet(fetty_little, fetty_little_name)
            print('writing file ',masky_little_name)
            write_mask(masky_little,masky_little_name)     
    return fetmask_dict              

#def run_subset_KK(kkdictobj, full_adjacency):
    #supercluster_info['kk_sub'][channel]
#    for channel in full_adjacency.keys():
#        kkdictobj['kk_sub'][channel].cluster_mask_starts()
#        superclusters[kkobj['sub_spikes'][channel],channel] = supercluster_info['kk_sub'][channel].clusters

def run_subset_KK(kkobj):
    #supercluster_info['kk_sub'][channel]
    kkobj.cluster_mask_starts()
    superclust_par = kkobj.clusters  
    return superclust_par

def run_subset_random_KK(kkobj):
    #supercluster_info['kk_sub'][channel]
    kkobj.cluster_random_starts()
    superclust_par = kkobj.clusters  
    return superclust_par

def run_subset_KK_chan(kkobject, channel):
    '''kkobject = supercluster_info['kk_sub']'''
    kkobject[channel].cluster_mask_starts()
    superclust_par = kkobject[channel].clusters  
    return superclust_par  

def squared(x):
    sq = x*x
    return sq

if __name__ == "__main__":
    scriptname = os.path.basename(__file__)
    print('Running script: ', os.path.basename(__file__))
    
    #sys.exit()
    basefolder = '/mnt/zserver/Data/multichanspikes/M140528_NS1/20141202/'
    littlename = '20141202_all'  
    basename =  basefolder + littlename   
    kwik_path = basename + '.kwik'

    model = KwikModel(kwik_path)
    #session = Session(kwik_path)

    #Make an old-fashioned .fet and .fmask file
    numb_spikes_to_use = 4000
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
    write_fet(fetty, fetbase)
    write_mask(triplemasky,fmaskbase)

    #sys.exit()

    #Make a full adjacency graph
    print('Making full adjacency graph')
    full_adjacency  = make_full_adjacency(model.probe.adjacency)
    print(full_adjacency)
    
    #active_channels = model.channels
    
    #Channel order dictionary
    channel_order_dict = make_channel_order_dict(model.channel_order)
    #embed()
    #For each channel find the indices of the points which are unmasked on this channel 
    print('Making globalcl_dict')   
    globalcl_dict = {}
    globalcl_dict = find_unmasked_points_for_channel(masky,channel_order_dict,full_adjacency,globalcl_dict)


    #compute indices of spike groups by taking the union of the unmasked indices
    #for each channel belonging to the spike group  
    globalcl_dict = find_unmasked_spikegroup(full_adjacency,globalcl_dict)
 
    #Make dictionary of subset features and masks 
    fetmask_dict = {}
    make_subset_fetmask(fetmask_dict, fetty, triplemasky, channel_order_dict, full_adjacency, globalcl_dict)    

    #Run MKK on each subset with the default value of the parameters
    script_params = default_parameters.copy()
    script_params.update(
    drop_last_n_features=0,
    save_clu_every=None,
    run_monitoring_server=False,
    save_all_clu=False,
    debug=True,
    start_from_clu=None,
    use_noise_cluster=True,
    use_mua_cluster=True,
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
    
    with open('%s_supercluster_info.p'%(derived_basename),'wb') as gg:
        pickle.dump(supercluster_info,gg)
    
    #Run KK2 on all the subsets
    numKK = len(full_adjacency.keys()) #equal to the number of channels
    superclusters = np.zeros((fetty.shape[0],numKK))
    c = Client(profile = 'default')
    lbv = c.load_balanced_view()
    lbv.block = True
    #with c[:].sync_imports():
    #    import klustakwik2 as *
    #c[:].execute('import klustakwik2 as *')
    c[:].execute('import klustakwik2')
    c[:].execute('from klustakwik2 import clustering')
    c[:]['supercluster_info']  =   supercluster_info
    c[:]['full_adjacency'] = full_adjacency
    #v = c[:]
    #v.map()
    start_time2 = time.time()
    supercluster_results = lbv.map(lambda channel: supercluster_info['kk_sub'][channel].cluster_mask_starts(),full_adjacency.keys())
    #supercluster_results = lbv.map(lambda channel: run_subset_KK(supercluster_info['kk_sub'][channel]),full_adjacency.keys())
    print('Time taken for parallel clustering %.2f s' %(time.time()-start_time2))
    #for channel in full_adjacency.keys():
    #    supercluster_info['kk_sub'][channel].cluster_mask_starts()
    #    supercluster_info['kk_sub'][channel].cluster_mask_starts()
    #    superclusters[supercluster_info['sub_spikes'][channel],channel] = supercluster_info['kk_sub'][channel].clusters
    
 
    # for channel in full_adjacency.keys():
    #     scriptname = basename+'%g'%(channel)
    #     scriptstring = '''import klustakwik2 as *
    #               '''
    #     scriptfile = open('%s.sh' %(scriptname),'w')
    #     scriptfile.write(scriptstring)
    #     scriptfile.close()
    #     changeperms='chmod 777 %s.sh' %(scriptname)
    #     os.system(changeperms)   
    
    #superinfo = [full_adjacency,globalcl_dict,supercluster_info,superclusters]
    print(supercluster_results)
    superinfo = [supercluster_results]
    with open('%s_supercluster.p'%(derived_basename), 'wb') as g:
        pickle.dump(superinfo, g)    
    
    
    
    
