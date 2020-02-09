import os, util

wins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SweetNEisenberg = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
residues = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

def getSeq(path):
    seq = ''
    for line in open(path, 'r'):
        if line[0] != '>':
            seq += line.strip()
    return seq

def slide_window(seq):

    ret = []
    for ind, res in enumerate(seq):
        vec = []
        for win in wins:
            start = max((ind-win), 0)
            end = min((ind+win+1), len(seq))
            sum = 0
            for j in xrange(start, end):
                r = seq[j]
                if not r in residues:
                    continue
                index = residues.index(r)
                sum += SweetNEisenberg[index]

            vec.append(sum)
        ret.append(vec)
    return ret


def getACH(path):
    ids = []
    with open(path + 'ID.txt', 'r') as file:
        ids.extend(file.readlines())
    ids = [id.strip() for id in ids]

    for ind, id in enumerate(ids):
        print id
        if not os.path.exists(util.vec_dir+'ach/'+id+'.txt'):
            seq = getSeq(path+'chain/'+id+'.txt')
            feature_vec = slide_window(seq)
            with open(util.vec_dir+'ach/'+id+'.txt', 'w') as out:
                for vec in feature_vec:
                    vec_str = [str(v) for v in vec]
                    out.write(' '.join(vec_str))
                    out.write('\n')

def getACH_(ids, seqs):
    for ind, id in enumerate(ids):
        print ind
        if not os.path.exists(util.vec_dir+'ach/'+id+'.txt'):
            feature_vec = slide_window(seqs[ind])
            with open(util.vec_dir+'ach/'+id+'.txt', 'w') as out:
                for vec in feature_vec:
                    vec_str = [str(v) for v in vec]
                    out.write(' '.join(vec_str))
                    out.write('\n')

def getACH_individual_(id, seq):
    feature_vec = slide_window(seq)
    with open(util.vec_dir+'ach/'+id+'.txt', 'w') as out:
        for vec in feature_vec:
            vec_str = [str(v) for v in vec]
            out.write(' '.join(vec_str))
            out.write('\n')

def main():
    getACH('../../PTM/DATA/P.ELM/Y/SRC/')

if __name__ == '__main__':
    main()
