import numpy as np
import ptm_features as pf
import cross_validation
from prep_vec import residue2vec
import prep_vec.util as util


prot2vec_window_size = 7
ST_kinases=['AGC_PKA', 'AGC_PKC', 'CGMC_CDK', 'Other_CK2']
Y_kinases =['TK_Src']


def writeLines(path, list):
    with open(path, 'w') as output:
        output.write('\n'.join(list))


def readLines(path):
    ret = []
    with open(path, 'r') as f:
        ret.extend(f.readlines())
    ret = [line.strip() for line in ret]
    return ret


def read_features(id):
    vec = pf.read_vecs(util.vec_dir + 'shannon/' + id + '.sh_vec')
    vec=pf.combine(vec, pf.read_vecs(util.vec_dir + 'relative/' + id + '.se_vec'))
    vec=pf.combine(vec, pf.read_vecs(util.vec_dir + 'pss/' + id + '.ss_vec'))
    vec=pf.combine(vec, pf.read_vecs(util.vec_dir + 'idr/' + id + '.idr'))
    vec=pf.combine(vec, pf.read_vecs(util.vec_dir + 'ach/' + id + '.txt'))
    vec=pf.combine(vec, pf.read_vecs(util.vec_dir + 'ops/' + id + '.txt'))

    return vec

def get_phospho_features(dir, win_size, sites):

    ids = pf.read_ids(dir)
    vecs = []
    ids, aa, labels = pf.read_data(dir, ids)
    for ind, id in enumerate(ids):
        assert (len(aa[ind]) == len(labels[ind]))

    new_ids = []
    for ind, id in enumerate(ids):
        # if id == 'A2ASS6' or id == 'Q8WZB3':
        #     continue
        try:
            vec = read_features(id)
        except:
            print id
            status = residue2vec.get_feature_(id, aa[ind])
            if status < 1:
                continue
            vec = read_features(id)

        # print id, '1st load finished. '
        if len(aa[ind]) != len(vec):
            status = residue2vec.get_feature_(id, aa[ind])
            if status < 1:
                continue
            vec = read_features(id)

        assert (len(aa[ind]) == len(vec))
        vecs.append(np.array(vec))
        new_ids.append(id)
        print ind

    with open(dir+'ID.txt', 'w') as output:
        output.write('\n'.join(new_ids))

    ids, aa, labels = pf.read_data(dir, new_ids)

    prot2vec_map, inferred_ids, inferred_vecs, pv_model = None, None, None, None


    aa, labels, new_vecs = pf.filter(aa, labels, vecs, win_size, sites,
                                     prot2vec_map=prot2vec_map,
                                     inferred_ids=inferred_ids,
                                     inferred_vecs=inferred_vecs,
                                     pv_model=pv_model)

    # windows, remains = pf.read_windows(aa, window=win_size, test=is_test, type=sites)
    return new_vecs, labels, aa, ids

class InputData:
    def __init__(self, dir, window_size):
        self._win_size = window_size

        features, labels, seqs, ids = get_phospho_features(dir,
                                                           win_size=self._win_size,
                                                           sites=['S', 'T', 'Y'])
        features = np.nan_to_num(features)

        positive_num = 0
        postives, negatives = [], []
        for ind, label in enumerate(labels):
            if label == 1:
                positive_num += 1
                postives.append(features[ind])
            else:
                negatives.append(features[ind])
        print len(postives), len(negatives)
        postives = np.array(postives)
        positive_labels = np.array([[0, 1]]*positive_num)
        negative_num = positive_num

        np.random.seed(1)
        perm = np.arange(len(negatives))
        np.random.shuffle(perm)


        negatives = np.array(negatives)[perm]
        negatives_selected = negatives[:negative_num]
        negative_labels = np.array([[1, 0]]*negative_num)

        self._extra_negative_features, self._extra_negative_labels = [], []
        extra_negative_num = int(negative_num*0.9)
        for i in xrange(0, len(negatives), extra_negative_num):
            end = min(len(negatives), i+extra_negative_num)
            if i+extra_negative_num <= len(negatives):
                self._extra_negative_features.append(negatives[i: end])
                self._extra_negative_labels.append([[1, 0]]*(end-i))

        features = np.concatenate([postives, negatives_selected], axis=0)
        labels = np.concatenate([positive_labels, negative_labels], axis=0)

        perm = np.arange(len(features))
        np.random.shuffle(perm)

        features = features[perm]
        labels = labels[perm]

        id_per_fold = len(labels)//10

        self._cv_feature_folds = []
        self._cv_labels_folds = []

        for i in xrange(10):
            if i == 9:
                self._cv_feature_folds.append(features[i*id_per_fold:])
                self._cv_labels_folds.append(labels[i * id_per_fold:])
            else:
                self._cv_feature_folds.append(features[i*id_per_fold: (i+1)*id_per_fold])
                self._cv_labels_folds.append(labels[i*id_per_fold: (i+1)*id_per_fold])


    def getData(self):
        return self._cv_feature_folds, self._cv_labels_folds, self._extra_negative_features, self._extra_negative_labels


class TestData:
    def __init__(self, dir, site, window_size):

        self._win_size = window_size

        features, labels, self._seqs, self._ids = get_phospho_features(dir,
                                                                       win_size=self._win_size,
                                                                       sites=site)
        aas = []
        for ind, seq in enumerate(self._seqs):
            # if ind < 10:
            for aa in seq:
                aas.append(aa)
        features_site, labels_site = [], []
        for ind, feature in enumerate(features):
            if aas[ind] in site:
                features_site.append(feature)
                if labels[ind] == 1:
                    labels_site.append([0, 1])
                else:
                    labels_site.append([1, 0])
        features_site = np.array(features_site)
        labels_site = np.array(labels_site)

        num_per_fold = len(features_site) // 10

        self._feature_folds = []
        self._labels_folds = []

        for i in xrange(10):
            if i == 9:
                self._feature_folds.append(features_site[i*num_per_fold:])
                self._labels_folds.append(labels_site[i * num_per_fold:])
            else:
                self._feature_folds.append(features_site[i* num_per_fold: (i+1)*num_per_fold])
                self._labels_folds.append(labels_site[i*num_per_fold: (i+1)*num_per_fold])


    def getData(self):
        return self._ids, self._seqs, self._feature_folds, self._labels_folds


def get_dataset(dir, win_size=5, is_negful=True):
    input_data = InputData(dir, window_size=win_size)
    cv_features, cv_labels, extra_negatives, extra_negative_labels = input_data.getData()
    if is_negful:
        return cross_validation.CrossValidation(cv_features, cv_labels, extra_negatives=extra_negatives, extra_labels=extra_negative_labels)
    else:
        return cross_validation.CrossValidation(cv_features, cv_labels)



def get_test_dateset(dir, site='Y', win_size=5, is_context=False):
    input_data = TestData(dir, site, window_size=win_size, is_context=is_context)
    ids, seqs, feature, label = input_data.getData()
    return ids, seqs, cross_validation.CrossValidation(feature, label), np.concatenate(label, axis=0)


def main():
    get_dataset('../DATA/Combined_train/Y/', 9, is_negful=False)


if __name__ == '__main__':
    main()
