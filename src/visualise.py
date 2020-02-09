import measures
from prep_vec import util
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Arial"

kinase_or_species = 0

netphos_kinase_map = dict()
netphos_kinase_map['CK2'] = 'CKII'
netphos_kinase_map['CDK'] = 'cdk5'
netphos_kinase_map['PKA'] = 'PKA'
netphos_kinase_map['PKC'] = 'PKC'
netphos_kinase_map['Src'] = 'SRC'

ST_kinases = ['AGC/PKC/Alpha/PKCA', 'AGC/PKC/Alpha/PKCB', 'AGC/PKC/Eta/PKCE',
              'AGC/PKC/Iota/PKCZ', 'AGC/PKC/Delta/PKCD']
Y_kinases = ['TK/Src']

def initialization(dir, type_name):
    phospho_type = dir + type_name

    candidates = dict()
    golds = dict()
    for line in open(phospho_type+'/ID.txt', 'r'):
        prot_id = line.strip()
        if prot_id != '':
            seq = util.getSeq(phospho_type+'/chain/'+prot_id+'.txt')
            labels = util.readLines(phospho_type+'/site/'+prot_id+'.txt')
            targets = []
            target_labels = []
            if kinase_or_species == 0:
                k = ['S', 'T']
                if type_name.startswith('TK/'):
                    k = ['Y']
                for ind, aa in enumerate(seq):
                    if aa in k:
                        targets.append(ind+1)
                        target_labels.append(int(labels[ind]))
            else:
                for ind, aa in enumerate(seq):
                    if aa == type_name[-1]:
                        targets.append(ind+1)
                        target_labels.append(int(labels[ind]))
            candidates[prot_id] = targets
            golds[prot_id] = target_labels
    return phospho_type, candidates, golds

def build_gold_predict_map(scores, golds):
    ret_golds = []
    ret_scores = []
    ret_ids = []
    ret_positions = []
    for id in golds:
        gold = golds[id]
        score = scores[id]
        assert(len(gold)==len(score))
        ret_ids.extend([id]*len(score))
        ret_positions.extend(xrange(len(score)))
        ret_golds.extend(gold)
        ret_scores.extend(score)
    return ret_ids, ret_positions, ret_golds, ret_scores

def get_transfer_aucs(type_name):
    golds = np.load('../FINALS/predicts/'+type_name+'.labels.npy')
    predicts = np.load('../FINALS/predicts/'+type_name+'.predicts.npy')
    predicts = np.nan_to_num(predicts)
    return None, None, golds, predicts

def get_predicts_gps_single_kinase(path, type_name, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [0]*len(candidates[candidate])
    id = ''
    for ind, line in enumerate(open(path, 'r')):
        if ind == 0:
            continue
        if line.startswith('>'):
            id = line.strip()[1:].strip()
        else:
            comps = line.strip().split('\t')
            position = int(comps[0])
            code = comps[1]
            t = ['S', 'T']
            if type_name.startswith('TK/'):
                t = ['Y']
            if code in t:
                if id in candidates and position in candidates[id] and float(comps[4]) > scores[id][candidates[id].index(position)]:
                    scores[id][candidates[id].index(position)] = float(comps[4])
    return build_gold_predict_map(scores, golds)

def get_predicts_gps(path, type_name, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [0]*len(candidates[candidate])
    id = ''
    for ind, line in enumerate(open(path+'/results/gps', 'r')):
        if ind == 0:
            continue
        if line.startswith('>'):
            id = line.strip()[1:].strip()
        else:
            comps = line.strip().split('\t')
            position = int(comps[0])
            code = comps[1]
            if kinase_or_species == 0:
                t = ['S', 'T']
                if type_name in Y_kinases:
                    t = ['Y']
                if code in t:
                    if id in candidates and position in candidates[id] and float(comps[4]) > scores[id][candidates[id].index(position)]:
                        scores[id][candidates[id].index(position)] = float(comps[4])
            else:
                t = type_name[-1]
                if code == t:
                    if id == 'AT1G24490.1':
                        print ''
                    if id in candidates and position in candidates[id] and float(comps[4]) > scores[id][
                        candidates[id].index(position)]:
                        scores[id][candidates[id].index(position)] = float(comps[4])
    return build_gold_predict_map(scores, golds)

def get_predicts_musite(path, type_name, name, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [-10]*len(candidates[candidate])
    id = ''
    for ind, line in enumerate(open(path+'/results/musite', 'r')):
        if ind == 0:
            continue
        if line.startswith('>'):
            id = line.strip()[1:]
        else:
            comps = line.strip().split('\t')
            position = int(comps[0])
            code = comps[1]
            if kinase_or_species == 0:
                if type_name in ST_kinases:
                    k = ['S', 'T']
                else:
                    k = ['Y']
                if code in k:
                    if position in candidates[id] and float(comps[3]) > scores[id][candidates[id].index(position)]:
                        scores[id][candidates[id].index(position)] = float(comps[3])
            else:
                if position in candidates[id] and float(comps[3]) > scores[id][candidates[id].index(position)]:
                    scores[id][candidates[id].index(position)] = float(comps[3])
    return build_gold_predict_map(scores, golds)

def get_predicts_phosphopick(path, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [0] * len(candidates[candidate])
    if not os.path.exists(path + '/results/phosphopick'):
        return build_gold_predict_map(scores, golds)
    for ind, line in enumerate(open(path + '/results/phosphopick', 'r')):
        if ind == 0:
            continue
        comps = line.strip().split('\t')
        protein_id = comps[0]
        position = int(comps[5])
        if comps[-1] == '-1':
            continue
        score = float(comps[-1])
        if position in candidates[protein_id] and score > scores[protein_id][candidates[protein_id].index(position)]:
            scores[protein_id][candidates[protein_id].index(position)] = score
    return build_gold_predict_map(scores, golds)


def get_predicts_kinasephos(path, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [0] * len(candidates[candidate])
    if not os.path.exists(path+'/results/kinasephos'):
        return build_gold_predict_map(scores, golds)
    id = ''
    position = -1
    new_protein = False
    recording = False
    count = 0
    for line in open(path+'/results/kinasephos', 'r'):
        if line.strip().startswith('Summary Result'):
            recording = False
            new_protein = True
            count = 0
            continue
        if new_protein:
            count += 1
            if count == 3:
                id = line.strip().split()[0]
                new_protein = False
                recording = True
            continue
        if recording:
            if line.strip() == 'There are no any phosphorylation site been predicted!!':
                recording = False
                new_protein = True
            if line.strip().isdigit():
                position = int(line.strip())
            elif '.' in line.strip():
                if not id in candidates:
                    print 'OPPS'
                    continue
                if position in candidates[id] and float(line.strip()) > scores[id][candidates[id].index(position)]:
                    scores[id][candidates[id].index(position)] = float(line.strip())
    return build_gold_predict_map(scores, golds)

def get_predicts_netphos(path, type_name, name, candidates, golds):
    scores = dict()
    for candidate in candidates:
        scores[candidate] = [0] * len(candidates[candidate])
    if not os.path.exists(path+'/results/netphos'):
        return build_gold_predict_map(scores, golds)
    id = ''
    for line in open(path+'/results/netphos', 'r'):
        if line.startswith('##Type Protein'):
            id = line.strip().split()[-1]
        elif not line.startswith('#'):
            splits = line.strip().split()
            position = int(splits[3])
            code = splits[2]
            comps = type_name.split('/')
            comps1 = code.split('-')
            if kinase_or_species == 1 or (kinase_or_species == 0 and (comps1[1] == netphos_kinase_map[comps[-1]])):
                if id in candidates and float(splits[5]) > scores[id][candidates[id].index(position)]:
                    scores[id][candidates[id].index(position)] = float(splits[5])
    return build_gold_predict_map(scores, golds)

def get_predict_phosphosvm(path):
    golds = np.load(path+'labels_noctxt.npy')
    predicts = np.load(path+'predicts_noctxt.npy')
    predicts = np.nan_to_num(predicts)
    return None, None, golds, predicts[:,1]

def get_predict_prot2vec(path):
    golds = np.load(path+'/results/labels.npy')
    predicts = np.load(path+'/results/predicts.npy')
    predicts = np.nan_to_num(predicts)
    return None, None, golds, predicts[:,1]

def get_predicts_results(path, type_name, phospho_type, candidates, golds):
    if path.endswith('gps'):
        return get_predicts_gps(phospho_type, type_name, candidates, golds)
    elif path.endswith('musite'):
        return get_predicts_musite(phospho_type, type_name,  path, candidates, golds)
    elif path.endswith('netphos'):
        return get_predicts_netphos(phospho_type, type_name, path, candidates, golds)
    elif path.endswith('kinasephos'):
        return get_predicts_kinasephos(phospho_type, candidates, golds)
    elif path.endswith('phosphopick'):
        return get_predicts_phosphopick(phospho_type, candidates, golds)
    elif path.endswith('noctxt'):
        return get_predict_phosphosvm(path[:-6])
    else:
        return get_predict_prot2vec(path)

def main(path, phospho_type, candidates, golds):
    if kinase_or_species == 0:
        dir = '../DATA/Combined_test/'
    else:
        dir = '../DATA/'
    systems = ['GPS_3.0']
    # output = open('../OUTPUT/results_kinase.txt', 'w')
    if kinase_or_species == 0:
        system_paths = [dir +path + '/results/gps']
        colors = ['darkorange']
        line_styles = ['-']
    else:
        systems = ['GPS_3.0', 'Musite_1.0', 'NetPhos_3.1', 'PhosphoProt2vec']
        system_paths = [dir + path + '/results/gps',
                        dir + path + '/results/musite',
                        dir + path + '/results/netphos',
                        dir + path]
        colors = ['darkorange', 'yellow', 'green',  'navy']
        line_styles = ['-', '-', '-', '-']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    o_str=''
    for ind, system in enumerate(systems):
        print system
        ret_ids, ret_positions, ret_golds, ret_scores = get_predicts_results(system_paths[ind], path, phospho_type, candidates, golds)
        # print system, len(ret_scores)
        # for ind1, id in enumerate(ret_ids):
        #     print path, system, id, ret_positions[ind1], ret_golds[ind1], ret_scores[ind1]
        #     output.write(path+','+system+','+id+','+str(ret_positions[ind1])+','+str(ret_golds[ind1])+','+str(ret_scores[ind1])+'\n')
        fpr[system], tpr[system], _, roc_auc[system], cutoff = measures.auc(ret_golds, ret_scores)
        pr, se, sp = measures.count(ret_golds, ret_scores, cutoff)
        o_str += "{0:.1f}".format(se * 100) + ' ' + "{0:.1f}".format(sp * 100) + ' ' + "{0:.1f}".format(
            measures.mcc(ret_golds, ret_scores, cutoff) * 100) + ' '
    print o_str

    # plt.figure()
    lw = 3
    for ind, system in enumerate(systems):
        if roc_auc[system] == 0.5:
            plt.plot(fpr[system], tpr[system], color='white', linestyle=line_styles[ind], lw=lw,
                     label=system + ' N/A ')
        else:
            plt.plot(fpr[system], tpr[system], color=colors[ind], linestyle=line_styles[ind], lw=lw,
                     label=system + ' (AUC = %0.3f)' % roc_auc[system])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curves for ' + path, fontsize=20)
    plt.legend(loc="lower right", prop={'size': 12})
    plt.grid(linestyle='--')
    plt.plot([-0.05, 1.05], [-0.05, 1.05], '--', color='grey', alpha=0.6)

def roc_plot():
    paths= ['AGC/PKC/Alpha/PKCA', 'AGC/PKC/Alpha/PKCB', 'AGC/PKC/Eta/PKCE', 'AGC/PKC/Iota/PKCZ', 'AGC/PKC/Delta/PKCD']
    dir = '../DATA/Combined_test/'
    # paths = ['PPA/S', 'PPA/T', 'PPA/Y']
    # dir = '../DATA/'

    if kinase_or_species == 0:
        plt.figure(figsize=(18, 12))
    else:
        plt.figure(figsize=(18, 6))

    for ind, path in enumerate(paths):
        print ind
        if kinase_or_species == 0:
            plt.subplot(2, 3, ind + 1)
        else:
            plt.subplot(1, 3, ind + 1)
        phospho_type, candidates, golds = initialization(dir, path)
        main(path, phospho_type, candidates, golds)
    plt.tight_layout()
    if kinase_or_species == 0:
        plt.savefig('kinase-specific.png', dpi=350)
    else:
        plt.savefig('general.png', dpi=350)

def auc_comparison():
    paths=[]
    for file in os.listdir('../FINALS/deepPhospho_results_GPS'):
        if file.endswith('.gps.fasta'):
            paths.append(file[:-10].replace('_', '/'))

    gps_aucs = []
    transfer_aucs = []
    for path in paths:
        phospho_type, candidates, golds = initialization('../DATA/Combined_test/', path)
        ret_ids, ret_positions, ret_golds, ret_scores = get_predicts_gps_single_kinase(
            '../FINALS/deepPhospho_results_GPS/'+path.replace('/', '_')+'.gps.fasta',
            path, candidates, golds)
        _, _, _, roc_auc, cutoff = measures.auc(ret_golds, ret_scores)
        gps_aucs.append(roc_auc)
        _, _, golds, scores = get_transfer_aucs(path.replace('/', '_'))
        _, _, _, roc_auc, cutoff = measures.auc(golds, scores)
        transfer_aucs.append(roc_auc)

    # Print number of better performance respectively
    transfer_larger, gps_larger = 0, 0
    for ind, path in enumerate(paths):
        if not path.startswith('TK/'):
            if transfer_aucs[ind] > gps_aucs[ind]:
                print path, transfer_aucs[ind], gps_aucs[ind], '\n'
                transfer_larger += 1
            elif transfer_aucs[ind] < gps_aucs[ind]:
                # print path, transfer_aucs[ind], gps_aucs[ind]
                gps_larger += 1

    print transfer_larger, gps_larger

    # Bar plot for performance difference
    diff=[]
    x_label = []
    path_diff = []
    for ind, path in enumerate(paths):
        if not path.startswith('TK'):
            x_label.append(path)
            diff.append(float(transfer_aucs[ind])-float(gps_aucs[ind]))
            path_diff.append((path, float(transfer_aucs[ind])-float(gps_aucs[ind])))
    x_pos = np.arange(len(x_label))

    sorted_path_diff = sorted(path_diff, key=lambda path_d: path_d[1])

    diff=[]
    x_label = []
    colors = []
    group = ['AGC', 'CK1', 'CAMK', 'Other', 'CMGC', 'Atypical', 'STE']
    color_groups = ['r', 'b', 'g', 'y', 'purple', 'grey', 'black', 'brown', 'while']
    for item in sorted_path_diff:
        diff.append(item[1])
        x_label.append(item[0])
        # if item[1] >=0:
        #     colors.append('red')
        # else:
        #     colors.append('blue')
        colors.append(color_groups[group.index(item[0][:item[0].index('/')])])
    plt.figure(figsize=(12, 6))
    plt.bar(x_pos, diff, align='center', alpha=0.5, color=colors, width=1)
    plt.xticks(x_pos, x_label, rotation='vertical')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # roc_plot()
    # pca_analysis()
    auc_comparison()
