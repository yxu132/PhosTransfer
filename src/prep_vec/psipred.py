import os, util
from collections import defaultdict

d = defaultdict(list)
d['H'].extend(['1', '0', '0'])
d['E'].extend(['0', '1', '0'])
d['C'].extend(['0', '0', '1'])

def readPSSHoriz(path):
    ss = ''
    for line in open(path, 'r'):
        if line.startswith('Pred'):
            ss += line.strip()[6:]
    return ss

def predict_secondary_structure_individual_(id):
    # print 'PSS:'+id
    os.system(util.tool_dir+'psipred-psipred-5f68ee3/runpsipred_ind'
        +' '+util.tool_dir+'psipred-psipred-5f68ee3'
        +' '+util.uniref90_psi_blast_database
        +' '+id
        +' '+util.chk_dir
        +' '+util.vec_dir+'ss/'
        +' '+os.path.abspath('.'))
    if not os.path.exists(util.vec_dir + 'ss/' + id + '.horiz'):
        return -3
    ss = readPSSHoriz(util.vec_dir+'ss/'+id+'.horiz')
    with open(util.vec_dir+'pss/'+id+'.ss_vec', 'w') as o:
        for s in ss:
            l = ' '.join(d[s]) + '\n'
            o.write(l)
    return 1

def predict_secondary_structure(path):
    ids = []
    with open(path+'ID.txt', 'r') as file:
        ids.extend(file.readlines())
    ids = [id.strip() for id in ids]

    for ind, id in enumerate(ids):
        print ind
        if not os.path.exists(util.vec_dir+'ss/'+id+'.horiz'):
            os.system(util.tool_dir+'psipred-psipred-5f68ee3/runpsipred '
                  + id
                  +' '+util.chk_dir
                  +' '+util.vec_dir+'ss/')
        if not os.path.exists(util.vec_dir+'pss/'+id+'.ss_vec'):
            ss = readPSSHoriz(util.vec_dir+'ss/'+id+'.horiz')
            with open(util.vec_dir+'pss/'+id+'.ss_vec', 'w') as o:
                for s in ss:
                    l = ' '.join(d[s]) + '\n'
                    o.write(l)

def predict_secondary_structure_(ids):
    for ind, id in enumerate(ids):
        print ind
        if not os.path.exists(util.vec_dir+'ss/'+id+'.horiz'):
            os.system(util.tool_dir+'psipred-psipred-5f68ee3/runpsipred'
                  +' '+util.tool_dir+'psipred-psipred-5f68ee3'
                  +' '+util.uniref90_psi_blast_database
                  +' '+id
                  +' '+util.chk_dir
                  +' '+util.vec_dir+'ss/'
                  +' '+os.path.abspath('.'))
        if not os.path.exists(util.vec_dir+'pss/'+id+'.ss_vec'):
            ss = readPSSHoriz(util.vec_dir+'ss/'+id+'.horiz')
            with open(util.vec_dir+'pss/'+id+'.ss_vec', 'w') as o:
                for s in ss:
                    l = ' '.join(d[s]) + '\n'
                    o.write(l)

def main():
    # predict_secondary_structure('../PTM/DATA/Uniprot/S/ATM/')
    predict_secondary_structure_individual_('O75151')

if __name__ == '__main__':
    main()
