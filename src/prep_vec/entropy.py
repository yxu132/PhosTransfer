import numpy as np
import os
import re
import psiblast
import util


aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
background_frequencies = [0.074, 0.025, 0.054, 0.054, 0.047, 0.074, 0.026, 0.068, 0.099, 0.058, 0.025, 0.045, 0.039, 0.034, 0.052, 0.057, 0.051, 0.073, 0.013, 0.034]

def psiblast_parse(path):
    ret = []
    residues = []
    for ind, line in enumerate(open(path, 'r')):
        if ind == 0:
            continue
        splits = re.sub(' +', ' ', line.strip())
        splits = splits.split(' ')
        comps = splits[2:]
        ret.append(comps)
        residues.append(splits[1])
    return residues, ret

def fix(splits):
    if len(splits) < 44:
        for ind, split in enumerate(splits):
            if '-' in split:
                index = split.index('-')
                if index > 0:
                    comp1 = split[:index]
                    comp2 = split[index:]
                    splits[ind] = comp1
                    splits.insert(ind+1, comp2)
                else:
                    if '-' in split[1:]:
                        index = split[1:].index('-')+1
                        if index > 0:
                            comp1 = split[:index]
                            comp2 = split[index:]
                            splits[ind] = comp1
                            splits.insert(ind + 1, comp2)
        return fix(splits)
    else:
        return splits

def test(path):
    for ind, line in enumerate(open(path, 'r')):
        # print line
        if ind < 3:
            continue
        if line == '\n':
            break
        splits = re.sub(' +', ' ', line.strip())
        splits = splits.split(' ')
        if len(splits) == 22:
            return True
        else:
            return False

def parse(path):
    # print path
    residues = []
    wops = []
    for ind, line in enumerate(open(path, 'r')):
        if ind < 3:
            continue
        if line == '\n':
            break
        splits = re.sub(' +', ' ', line.strip())
        splits = splits.split(' ')
        # print splits
        fix(splits)
        comps = [int(w) for w in splits[22:42]]
        residues.append(line[6])
        wops.append(comps)
        # for comp in comps:
        #     print comp
    return residues, np.array(wops)

def shannon_entropy(path):
    _, matrix = parse(path)
    se_protein = []
    for array in matrix:
        p_sum = np.sum(array)
        p = array / float(p_sum)
        se = 0
        for p_i in p:
            se += p_i * np.log(p_i+0.00000001)
        se = -se
        se_protein.append(se)
    return se_protein

def relative_entropy(path):
    residues, matrix = parse(path)
    re_protein = []
    for ind, array in enumerate(matrix):
        p_sum = np.sum(array)
        p = array / float(p_sum)
        re = 0
        residue = residues[ind]
        if residue in aa:
            index = aa.index(residue)
            for p_i in p:
                re += p_i * np.log(p_i/background_frequencies[index]+0.00000001)
            re = -re
            re_protein.append(re)
        else:
            re_protein.append(0.0)
    return re_protein

def WOP(path):
    residues, matrix = parse(path)
    re_protein = []
    for ind, array in enumerate(matrix):
        p_sum = np.sum(array)
        parray = array / float(p_sum)
        array = [str(a) for a in parray]
        re_protein.append(array)
    return re_protein

def getEntropy(path):
    ids = []
    with open(path+'ID.txt', 'r') as file:
        ids.extend(file.readlines())
    ids = [id.strip() for id in ids]

    for ind, id in enumerate(ids):
        if not os.path.exists(util.vec_dir+'shannon/'+id+'.sh_vec'):
            print ind
            se = shannon_entropy(util.wop_dir+id+'.wop')
            re = relative_entropy(util.wop_dir+id+'.wop')
            with open(util.vec_dir+'shannon/'+id+'.sh_vec', 'w') as output:
                output.write('\n'.join([str(s) for s in se]))
            with open(util.vec_dir+'shannon/'+id+'.se_vec', 'w') as output:
                output.write('\n'.join([str(s) for s in re]))

def getEntropy_individual_(id, seq):
    ret = psiblast.getBlast_individual_(id, seq)
    if ret < 1:
        return ret
    se = shannon_entropy(util.wop_dir+id+'.wop')
    re = relative_entropy(util.wop_dir+id+'.wop')
    with open(util.vec_dir + 'shannon/' + id + '.sh_vec', 'w') as output:
        output.write('\n'.join([str(s) for s in se]))
    with open(util.vec_dir + 'relative/' + id + '.se_vec', 'w') as output:
        output.write('\n'.join([str(s) for s in re]))
    return 1

def getEntropy_(ids, seqs):
    for ind, id in enumerate(ids):
        if not os.path.exists(util.vec_dir+'shannon/'+id+'.sh_vec'):
            if not os.path.exists(util.wop_dir+id+'.wop'):
                psiblast.getBlast_(id, seqs[ind])
            se = shannon_entropy(util.wop_dir+id+'.wop')
            re = relative_entropy(util.wop_dir+id+'.wop')
            with open(util.vec_dir+'shannon/'+id+'.sh_vec', 'w') as output:
                output.write('\n'.join([str(s) for s in se]))
            with open(util.vec_dir+'relative/'+id+'.se_vec', 'w') as output:
                output.write('\n'.join([str(s) for s in re]))

if __name__ == '__main__':
    getEntropy('../../PTM/DATA/P.ELM/Y/SRC/')
