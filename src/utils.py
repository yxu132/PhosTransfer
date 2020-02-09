import numpy as np



def readLines(path):
    ret = []
    for line in open(path, 'r'):
        ret.append(line.strip())
    return ret

def readFasta(path, tsv=False):
    ids, seqs = [], []
    if tsv:
        for line in open(path, 'r'):
            comps = line.strip().split()
            ids.append(comps[0])
            seqs.append(comps[1])
    else:
        seq = ''
        id = ''
        for line in open(path, 'r'):
            if line.startswith('>'):
                if id != '':
                    ids.append(id)
                    seqs.append(seq)
                id = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
        if id != '':
            ids.append(id)
            seqs.append(seq)
    return ids, seqs

def read_vecs(path):
    vecs = []

    for line in open(path, 'r'):
        comps = line.strip().split()
        if len(comps) > 1:
            comps = [np.float32(c) for c in comps]
            vecs.append(comps)
        else:
            vecs.append([np.float32(line.strip())])

    return np.asarray(vecs)

def combine(vec1, vec2):
    ret = np.concatenate([vec1, vec2], axis=1)
    return ret

def readLabels(ids, dir):
    ret = []
    for id in ids:
        labels = readLines(dir+'/'+id+'.txt')
        ret.append(labels)
    return ret

def readFastas(ids, dir):
    ret = []
    for id in ids:
        for ind, line in enumerate(open(dir+'/'+id+'.txt', 'r')):
            if ind == 0 or line.strip() == '':
                continue
            ret.append(line.strip())
    return ret