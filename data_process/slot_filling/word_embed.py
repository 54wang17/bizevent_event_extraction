import numpy as np
import cPickle
from gensim.models import Word2Vec
import os
if __name__=="__main__":

    event = 'ipo'
    w2v_file = 'path/to/GoogleNews-vectors-negative300.bin'
    model = Word2Vec.load_word2vec_format(w2v_file, binary=True)
    embd_dim = 300
    groups = ['a', 'b', 'b1', 'b2']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace('data_process', 'data')
    for group in groups:
        y = cPickle.load(open(os.path.join(dir_path, event, "corpus_%s.p" % group), "rb"))
        word2idx, idx2word = y[6], y[7]
        assert len(idx2word) == len(word2idx)
        vocab_size = len(idx2word)
        W = np.zeros(shape=(vocab_size, embd_dim))
        for word in word2idx:
            if word in model.vocab:
                W[word2idx[word]] = model[word]
            else:
                W[word2idx[word]] = np.random.uniform(-0.25, 0.25, embd_dim)
        cPickle.dump([W], open(os.path.join(dir_path, event, "word2vec_%s.p" % group), "wb"))
    print "Pre-trained word vector created!"