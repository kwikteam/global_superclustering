# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import hashlib as hl
import joblib_utils as ju
import string
import collections as col



@ju.func_cache
def hash_dictionary_md5(dict1):
    hashdict = hl.md5(repr(sorted(dict1.items(), key= lambda (key,val): key))).hexdigest()
    print hashdict
    return hashdict

@ju.func_cache
def get_product_hashlist(dictlist):
    '''gives the product hash of a list of dictionaries,
       e.g. dictlist = [dict1, dict2, dict3]
       product_hash = [Hash(dict1),Hash(dict2),Hash(dict3)]
    '''
    dictlisthash = []
    for dictio in dictlist:
        dictlisthash.append(hash_dictionary_md5(dictio))
    return dictlisthash

@ju.func_cache
def make_concatenated_filename(dictlisthash):
    ''' concatenate [Hash(dict1),Hash(dict2),Hash(dict3)] to give
        producthash = Hash(dict1)_Hash(dict2)_Hash(dict3)'''
    producthash = string.join(dictlisthash,'_')
    return producthash

@ju.func_cache        
def make_SDoutputname(hybdatadict,sdparams):
    # Make the product hash output name
    hashSDparams = hash_dictionary_md5(sdparams)
    # chose whether to include the probe, if so uncomment the two lines below
    #hashprobe = hash_utils.hash_dictionary_md5(prb)
    #hashdictlist = [ hybdatadict['hashD'],hashSDparams, hashprobe]
    hashdictlist = [hybdatadict['hashD'],hashSDparams]
    hash_hyb_SD_prb = make_concatenated_filename(hashdictlist)
    return hash_hyb_SD_prb

    
def order_dictionary(dict1):
    '''Orders a dictionary by key'''
    orddict = col.OrderedDict(sorted(dict1.items(), key = lambda t:t[0]))
    return orddict


   
