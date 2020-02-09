import numpy as np
import gensim.models as model
import os
import urllib2

infer_alpha = 0.01
infer_epoch = 30000

def readLines(path):
    ret = []
    with open(path, 'r') as file:
        ret.extend(file.readlines())
    ret = [line.strip() for line in ret]
    return ret

def downloadSeq(prot_id):
    ret = ''
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

class Prot2vecModel:

    def __init__(self, input_model_path,
                 uniprot_id_path,
                 shuffle_id_path,
                 raw_path,
                 vector_type='fetch',
                 n_gram_model='overlap'):
        print input_model_path
        self._model = model.Doc2Vec.load(input_model_path)
        self._uniprot_ids = []
        for line in open(uniprot_id_path, 'r'):
            self._uniprot_ids.append(line.strip())
        self._shuffle_ids = []
        self._n_gram_model = n_gram_model
        if n_gram_model == 'overlap':
            for line in open(shuffle_id_path, 'r'):
                if line.strip() == '':
                    self._shuffle_ids.append(-1)
                else:
                    self._shuffle_ids.append(int(line.strip()))
        else:
            for i in xrange(3):
                shuffle_ids = []
                for line in open(shuffle_id_path+'_'+str(i)+'.txt', 'r'):
                    if line.strip() == '':
                        shuffle_ids.append(-1)
                    else:
                        shuffle_ids.append(int(line.strip()))
                self._shuffle_ids.append(shuffle_ids)
        self._seqs = []
        for line in open(raw_path, 'r'):
            self._seqs.append(line.strip().split('\t')[1])

        self._vector_type = vector_type  # 'add', 'fetch' or 'non_overlap_fetch'



    def ngram(self, seq):
        ret = []
        if self._n_gram_model == 'overlap':
            for i in range(0, len(seq) - 3):
                ret.append(seq[i: i + 3])
        else:
            seq_size = len(seq)
            for i in range(0, seq_size, 3):
                if i + 3 >= seq_size:
                    ret.append(seq[i:])
                else:
                    ret.append(seq[i:i + 3])
        return ret

    def infer(self, n_grams, alpha, steps):
        prot2vec = self._model.infer_vector(n_grams, alpha=alpha, steps=steps)
        return prot2vec

    def get(self, seq):
        n_grams = self.ngram(seq)
        # infer prot2vec for the sequence
        prot2vec = self.infer(n_grams, alpha=infer_alpha, steps=infer_epoch)
        return prot2vec

    def get_add(self, seq):
        word2vec = self._model.syn0
        words = self._model.index2word
        n_grams = self.ngram(seq)
        prot2vec = np.zeros(len(word2vec[0]))
        for ngram in n_grams:
            if ngram in words:
                index = words.index(ngram)
                wv = word2vec[index]
                prot2vec = np.add(prot2vec, wv)
        return prot2vec


    def fetch(self, prot_ids, prot_seqs=None):
        prot2vecs = []
        for ind, prot_id in enumerate(prot_ids):
            print('-----------infering prot2vec ' + str(ind) + '-------------' + '\n')
            # splits sequence into overlap 3-gram segments
            if prot_seqs != None:
                seq = prot_seqs[ind]
            elif prot_id in self._uniprot_ids:
                index = self._uniprot_ids.index(prot_id)
                seq = self._seqs[index]
            else:
                seq = downloadSeq(prot_id)
            n_grams = self.ngram(seq)
            # infer prot2vec for the sequence
            prot2vec = self.infer(n_grams, alpha=infer_alpha, steps=infer_epoch)
            # update the inferred_vectors to accelerate future runs
            print('-----------infering prot2vec '+str(ind)+'-------------' + '\n')
            prot2vecs.append(prot2vec)
        return prot2vecs

    def add(self, prot_ids, prot_seqs=None):
        prot2vecs = []
        word2vec = self._model.syn0
        words = self._model.index2word
        for ind, prot_id in enumerate(prot_ids):
            if prot_seqs != None:
                seq = prot_seqs[ind]
            elif prot_id in self._uniprot_ids:
                index = self._uniprot_ids.index(prot_id)
                seq = self._seqs[index]
            else:
                seq = downloadSeq(prot_id)
            n_grams = self.ngram(seq)
            prot2vec = np.zeros(len(word2vec[0]))
            for ngram in n_grams:
                if ngram in words:
                    index = words.index(ngram)
                    wv = word2vec[index]
                    prot2vec = np.add(prot2vec, wv)
            prot2vecs.append(prot2vec)
        return prot2vecs