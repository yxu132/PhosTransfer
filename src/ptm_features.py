import numpy as np
import os, math
from prep_vec import util

def read_vecs(path):
    vecs = []

    for line in open(path, 'r'):
        comps = line.strip().split()
        if len(comps) > 1:
            comps = [np.float32(c) for c in comps]
            vecs.append(comps)
        else:
            vecs.append([np.float32(line.strip())])

    return np.array(vecs)


def combine(vec1, vec2):
    ret = np.concatenate([vec1, vec2], axis=1)
    return ret

def read_ids(path):
    ids = []
    for ind, line in enumerate(open(path + 'ID.txt', 'r')):
        if not os.path.exists(util.chk_dir+line.strip()+'.chk'):
            continue
        ids.append(line.strip())
    return ids

def read_data(path, ids):

    aas = []
    for id in ids:
        if not os.path.exists(path+'chain/'+id+'.txt'):
            print id
        for line in open(path+'chain/'+id+'.txt', 'r'):
           if not line.startswith('>'):
               aas.append(line.strip())

    labels = []
    for id in ids:
        sites = []
        for line in open(path+'site/'+id+'.txt', 'r'):
            sites.append(int(line.strip()))

        labels.append(sites)

    return ids, aas, labels

def read_prot2vecs(path, ids_order):
    vecs = []
    ids = []
    for line in open(path+'prot2vec.txt', 'r'):
        id, comps = line.strip().split('\t')
        vec = comps.split(', ')
        vec = [float(v) for v in vec]
        vecs.append(vec)
        ids.append(id)
    ret = []
    for id in ids_order:
        ret.append(vecs[ids.index(id)])
    return ret

def read_windows(aas, window, test=False, type=None):
    ret = []
    remains = []
    for aa in aas:
        remain = []
        # vec = []
        for ind in xrange(len(aa)):
            if test:
                if aa[ind] in type:
                    # start = max([0, ind-window])
                    # end = min([len(aa), ind+window+1])
                    # vec.append(aa[start:end])
                    remain.append(ind)
            else:
                if aa[ind] == 'S' or aa[ind] == 'Y' or aa[ind] == 'T':
                    # start = max([0, ind-window])
                    # end = min([len(aa), ind+window+1])
                    # vec.append(aa[start:end])
                    remain.append(ind)
        # ret.append(vec)
        remains.append(remain)
    return np.array(ret), remains

def filter(aas, labels, vecs, win_size, sites, prot2vec_map=None,
           inferred_ids=None, inferred_vecs=None,
           pv_model=None):
    # print len(prot2vec_map)

    n_aas, n_labels, n_vecs = [],  [], []
    for indi, aa in enumerate(aas):
        # print indi
        for indj in xrange(len(aa)):
            if aa[indj] in sites:
                n_labels.append(labels[indi][indj])
                n_aas.append(aa[indj])
                vals = []
                for j in xrange(indj - win_size, indj + win_size + 1):
                    if j < 0:
                        vals.extend(np.zeros(26))
                    elif j >= len(aa):
                        vals.extend(np.zeros(26))
                    else:
                        vals.extend(vecs[indi][indj])
                n_vecs.append(vals)

    return n_aas, n_labels, n_vecs

def filter_in_bulks(aas, vecs, win_size, sites):
    n_vecs = []
    for indi, aa in enumerate(aas):
        for indj in xrange(len(aa)):
            if aa[indj] in sites:
                vals = []
                for j in xrange(indj - win_size, indj + win_size + 1):
                    if j < 0:
                        vals.extend(np.zeros(26))
                    elif j >= len(aa):
                        vals.extend(np.zeros(26))
                    else:
                        vals.extend(vecs[indi][indj])
                n_vecs.append(vals)
    return n_vecs, vecs[2000:]
