import urllib2

home_path = '/media/ying/YING_backup/PHOSPHO'

vec_dir = home_path+'/vecs/'
wop_dir = home_path+'/wop/'
chk_dir = home_path+'/chk/'

uniref90_psi_blast_database = '/home/ying/mnt/data/uniref90filt/uniref90filt'

blast_path='/usr/bin/'

blast_database_dir = blast_path

tool_dir = '/home/ying/Tools/'


def readLines(file_name):
    ret = []
    with open(file_name, 'r') as input:
        ret.extend(input.readlines())
    ret = [r.strip() for r in ret]
    return ret

def input_str(input):
    ids, seqs = [], []
    input = input.strip()
    if '\n' in input:
        lines = input.split('\n')
        current_id, current_seq = '', ''
        for line in lines:
            if line.startswith('>'):
                if current_id != '':
                    ids.append(current_id)
                    seqs.append(current_seq)
                current_id = line.strip()[1:]
                current_id = current_id.split('|')[1]
                current_seq = ''
            else:
                current_seq += line.strip()
        if current_id != '':
            ids.append(current_id)
            seqs.append(current_seq)
        else:
            return ids, seqs
    else:
        if ', ' in input:
            ids = input.split(', ')
        elif ',' in input:
            ids = input.split(',')
        else:
            ids.append(input)
        for i in xrange(len(ids)):
            seqs.append('')        
    return ids, seqs

def downloadSeq(prot_id):
    ret = ''
    response = ''
    try:
        response = urllib2.urlopen('http://www.uniprot.org/uniprot/'+prot_id+'.fasta')
    except urllib2.HTTPError:
        # Do something here to handle the error. For example:
        print("URL", prot_id, "could not be read.")
        pass
    for line in response:
        if not line[0] == '>':
            ret += line.strip('\n')
        else:
            if len(ret) > 0:
                break
    return ret

def truncate_length(seq, limit):
    if len(seq) >= limit:
        seq = seq[:limit]
    return seq


def parse_results(result, ktype, ids, seqs, vecs, remains, above, threshold=0.5):
    ret = []
#    ret.append('# seqname \t position \t code \t kinase \t context \t score \t cutoff \t ?')
    if '_' in ktype:
        comps = ktype.split('_')
    else:
        comps = [ktype, '-']
    current_index = 0
    for indj, seq in enumerate(seqs):
	remain = remains[indj]
        num = len(remain)
        res = result[current_index: current_index+num]
        for indk, rem in enumerate(remain):
            if float(res[indk][1]) > above:
                if float(res[indk][1]) > threshold:
                    label = 'YES'
                else:
                    label = '..'
                ret.append(ids[indj]+'\t'+str(rem+1)+'\t'+comps[0]+'\t'+comps[1]+'\t'+vecs[indj][indk]+'\t'+"{:.3f}".format(res[indk][1])+'\t'+str(threshold)+'\t'+label)		
        current_index += num
    return ret

def parse_ids_seqs(ids, seqs):
    ret = []
    ret.append('# id-seq')
    for ind, id in enumerate(ids):
        ret.append(id+'\t'+seqs[ind])
    return ret

def readResults(str):
    lines = str.split('\n')
    res_map = dict()
    for ind, line in enumerate(lines):
        if ind == 0:
            continue
        if line.strip() == '# id-seq':
            break
        comps = line.strip().split('\t')
        if not comps[0] in res_map:
            res_map[comps[0]] = []
        res_map[comps[0]].append(comps[1:])
    seq_map = dict()
    start = False
    for ind, line in enumerate(lines):
        if line.strip() == '# id-seq':
            start = True
            continue
        if start:
            print line
            comps = line.strip().split('\t')
            seq_map[comps[0]] = comps[1]
    return res_map, seq_map

def visualResults(str):
    res_map, seq_map = readResults(str)
    vis_map = dict()
    for protein in res_map:
        maps = dict()
        seq = seq_map[protein]
        for comps in res_map[protein]:
            ktype = comps[1]
            if comps[2] != '-':
                ktype = comps[1]+'_'+comps[2]
            if not ktype in maps:
                maps[ktype] = [0.0] * len(seq)
            if comps[-1] == 'YES':
                maps[ktype][int(comps[0])-1] = float(comps[4])
        vis_map[protein] = maps
    return vis_map

def readLine(path):
    lines = []
    with open(path, 'r') as input:
        lines.extend(input.readlines())
    lines = [line.strip() for line in lines]
    return lines

def getSeq(path):
    lines = readLine(path)
    seq = ''.join(lines[1:])
    return seq

if __name__ == '__main__':
    input_str('P60983')
