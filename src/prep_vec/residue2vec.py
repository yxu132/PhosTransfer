import entropy
import psipred
import disordered
import ach
import op

def get_feature_(id, seq):
    ret = entropy.getEntropy_individual_(id, seq)
    if ret < 1:
        return ret
    ret = psipred.predict_secondary_structure_individual_(id)
    if ret < 1:
        return ret

    disordered.predict_disordered_individual_(id)

    ach.getACH_individual_(id, seq)
    op.getOP_individual_(id, seq)
    return id, seq


if __name__ == '__main__':

    path = 'ST'   # Path to proteins that need feature regeneration

    start, length = 10000, 1000
    count = 0
    print 'Start from:', start, ' End at: ', start+length, ' for site: ', path
    id_seq_map = dict()
    for line in open('../../DATA/phospho.fasta', 'r'):
        comps = line.strip().split()
        if len(comps) == 2:
            id_seq_map[comps[0]] = comps[1]

    redo_list = []
    for ind, line in enumerate(open('../../DATA/Combined_train/'+path+'/ID.txt', 'r')):
        print ind
        if ind < start:
            continue

        # if ind < start + length:
        id = line.strip()
        if id in id_seq_map:
            seq = id_seq_map[id]
            ret = get_feature_(id, seq=seq)
            if ret == -1:
                print 'fail blast at '+str(ind)+ ': '+id
            if ret == -2:
                redo_list.append(id)
                print 'redo psiblast required at'+str(ind)+ ': '+id
            if ret == -3:
                print 'failed to open chk file for psipred'

    print 'redo num: ', len(redo_list)
    print redo_list
    print count

