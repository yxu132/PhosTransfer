import os, util
from collections import defaultdict

d = defaultdict(int)
d['*'] = 1
d['.'] = 0

def readPSSHoriz(path):
    ss = ''
    skip = False
    for line in open(path, 'r'):
        if line.startswith('conf: ') and len(line.strip()) == 5:
            skip = True
        if line.startswith('pred: '):
            if not skip:
                ss += line.strip()[6:]
            else:
                skip = False
    return ss

def predict_secondary_structure(path):
    ids = []
    with open(path+'ID.txt', 'r') as file:
        ids.extend(file.readlines())
    ids = [id.strip() for id in ids]

    for ind, id in enumerate(ids):
        print ind
        if not os.path.exists(util.vec_dir+'disord/' + id + '.txt.horiz_d'):
        # if id == 'P15340':
            os.system(util.tool_dir+'DISOPRED/run_disopred.pl '
                  +id + ' '
                  +util.chk_dir+id+'.chk' + ' '
                  +util.vec_dir+'disord/')
        if not os.path.exists(util.vec_dir + 'disorder/' + id + '.disord_vec'):
            ss = readPSSHoriz(util.vec_dir+'disord/' + id + '.txt.horiz_d')
            with open(util.vec_dir + 'disorder/' + id + '.disord_vec', 'w') as o:
                for s in ss:
                    l = str(d[s]) + '\n'
                    o.write(l)

def predict_secondary_structure_(ids):
    for ind, id in enumerate(ids):
        print ind
        if not (os.path.exists(util.vec_dir+'disord/' + id + '.txt.horiz_d') or os.path.exists(util.vec_dir+'disord/' + id + '.horiz_d')):
            os.system(util.tool_dir+'DISOPRED/run_disopred.pl '
                  +util.blast_path + ' '
                  +util.tool_dir +'psipred-psipred-5f68ee3/bin/ '
                  +util.uniref90_psi_blast_database+' '
                  +id + ' '
                  +util.chk_dir+id+'.chk' + ' '
                  +util.vec_dir+'disord/')
        if not os.path.exists(util.vec_dir + 'disorder/' + id + '.disord_vec'):
    	    if os.path.exists(util.vec_dir+'disord/' + id + '.horiz_d'):
    	        ss = readPSSHoriz(util.vec_dir+'disord/' + id + '.horiz_d')
    	    elif os.path.exists(util.vec_dir+'disord/' + id + '.txt.horiz_d'):
    		    ss = readPSSHoriz(util.vec_dir+'disord/' + id + '.txt.horiz_d')
            with open(util.vec_dir + 'disorder/' + id + '.disord_vec', 'w') as o:
                for s in ss:
                    l = str(d[s]) + '\n'
                    o.write(l)

def predict_disordered_individual_(id):
    os.system(util.tool_dir+'DISOPRED/run_disopred_ind.pl '
                  +util.blast_path + ' '
                  +util.tool_dir +'psipred-psipred-5f68ee3/bin/ '
                  +util.uniref90_psi_blast_database+' '
                  +id + ' '
                  +util.chk_dir+id+'.chk' + ' '
                  +util.vec_dir+'disord/')
    ss = []
    if os.path.exists(util.vec_dir+'disord/' + id + '.horiz_d'):
        ss = readPSSHoriz(util.vec_dir+'disord/' + id + '.horiz_d')
    elif os.path.exists(util.vec_dir+'disord/' + id + '.txt.horiz_d'):
        ss = readPSSHoriz(util.vec_dir+'disord/' + id + '.txt.horiz_d')
    with open(util.vec_dir + 'disorder/' + id + '.disord_vec', 'w') as o:
        for s in ss:
            l = str(d[s]) + '\n'
            o.write(l)

def main():
    predict_secondary_structure('../PTM/DATA/P.ELM/S/PKC/')

if __name__ == '__main__':
    main()
