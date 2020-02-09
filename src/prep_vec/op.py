import os, util

ops = []
ops.append(['N', 'Q', 'S', 'D', 'E', 'C', 'T', 'K', 'R', 'H', 'Y', 'W'])
ops.append(['K', 'H', 'R'])
ops.append(['D', 'E'])
ops.append(['K', 'H', 'R', 'D', 'E'])
ops.append(['A', 'G', 'C', 'T', 'I', 'V', 'L', 'K', 'H', 'F', 'Y', 'W', 'M'])
ops.append(['I', 'V', 'L'])
ops.append(['F', 'Y', 'W', 'H'])
ops.append(['P', 'N', 'D', 'T', 'C', 'A', 'G', 'S', 'V'])
ops.append(['A', 'S', 'G', 'C'])
ops.append(['P'])

def getSeq(path):
    seq = ''
    for line in open(path, 'r'):
        if line[0] != '>':
            seq += line.strip()
    return seq


def getOP(path):
    ids = []
    with open(path + 'ID.txt', 'r') as file:
        ids.extend(file.readlines())
    ids = [id.strip() for id in ids]

    for ind, id in enumerate(ids):
        print id
        if not os.path.exists(util.vec_dir+'ops/'+id+'.txt'):
            seq = getSeq(path+'chain/'+id+'.txt')
            with open(util.vec_dir+'ops/'+id+'.txt', 'w') as out:
                for res in seq:
                    vec = []
                    for op in ops:
                        if res in op:
                            vec.append('1')
                        else:
                            vec.append('0')
                    out.write(' '.join(vec))
                    out.write('\n')

def getOP_(ids, seqs):
    for ind, id in enumerate(ids):
        print ind
        if not os.path.exists(util.vec_dir+'ops/'+id+'.txt'):
            with open(util.vec_dir+'ops/'+id+'.txt', 'w') as out:
                for res in seqs[ind]:
                    vec = []
                    for op in ops:
                        if res in op:
                            vec.append('1')
                        else:
                            vec.append('0')
                    out.write(' '.join(vec))
                    out.write('\n')

def getOP_individual_(id, seq):
    with open(util.vec_dir+'ops/'+id+'.txt', 'w') as out:
        for res in seq:
            vec = []
            for op in ops:
                if res in op:
                    vec.append('1')
                else:
                    vec.append('0')
            out.write(' '.join(vec))
            out.write('\n')

def main():
    getOP('../../PTM/DATA/P.ELM/Y/SRC/')

if __name__ == '__main__':
    main()
