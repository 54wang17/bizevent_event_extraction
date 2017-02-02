import os
from collections import defaultdict
import random
import cPickle


def load_file(file_path, file):
    l = []
    with open(os.path.join(file_path, file), 'rb') as f:
        for line in f:
            text = line.strip().lower()
            l.append(text)
    return l


def load_data(file_path='../data', train_file='train.txt', test_file='test.txt',
              train_label='train_label.txt', test_label='test_label.txt'):

    train = load_file(file_path, train_file)
    test = load_file(file_path, test_file)
    train_label = load_file(file_path, train_label)
    test_label = load_file(file_path, test_label)

    return train, test, train_label, test_label


def build_vocab(corpus, min_df=10):
    '''
    :param corpus:
    :param min_df:
    :return: index2word, word2index
    '''
    vocab = defaultdict(int)

    for sent in corpus:
        # words = set(sent.split())
        for word in sent.split():
            vocab[word] += 1

    index2word, word2index = {}, {}

    index2word[0] = '<UNKNOWN>'
    word2index['<UNKNOWN>'] = 0
    ix = 1
    for w in vocab:
        if vocab[w] < min_df:
            continue
        word2index[w] = ix
        index2word[ix] = w
        ix += 1
    del vocab

    return index2word, word2index


def transform_data(sentences, word2index):
    '''
    :param sentences: a list of sentence, each sentence is a string
    :param word2index: a map of word with its corresponding index
    :return: a 2D list, each row represents a sentence, each elem in row is an index representing a word
    '''
    data = []
    for rev in sentences:
        sent = get_idx_from_sent(rev, word2index)
        data.append(sent)

    return data


def get_idx_from_sent(sentence, word2index):
    '''
    :param sentence: a string of sentence
    :param word2index: map of word and index
    :return: a list of index
    '''
    words = sentence.split()
    index = [word2index[w] if w in word2index else word2index['<UNKNOWN>'] for w in words]

    return index


def label2category(label):
    index2label, label2index = build_vocab(label, min_df=0)
    del label2index['<UNKNOWN>'], index2label[0]
    for index in index2label:
        print 'Label {} -> Category id {}'.format(index2label[index], index)

    return label2index, index2label


def create_val_test(text_index, text, label_data):
    assert len(text_index) == len(text)
    n_data = len(text)
    print "text length: {}".format(n_data)

    n_valid = int(0.1 * n_data)
    valid_indices = random.sample(xrange(n_data), n_valid)
    train_text_index = [text_index[i] for i in xrange(n_data) if i not in valid_indices]
    valid_text_index = [text_index[i] for i in xrange(n_data) if i in valid_indices]
    train_text = [text[i] for i in xrange(n_data) if i not in valid_indices]
    valid_text = [text[i] for i in xrange(n_data) if i in valid_indices]
    train_label = [label_data[i] for i in xrange(n_data) if i not in valid_indices]
    valid_label = [label_data[i] for i in xrange(n_data) if i in valid_indices]

    print "train text length: {}, train label length: {}".format(len(train_text), len(train_label))
    print "val text length: {}, val label length: {}".format(len(valid_text), len(valid_label))

    return train_text_index, valid_text_index, train_text, valid_text, train_label, valid_label

if __name__ == "__main__":
    groups = ['a']
    for group in groups:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path.replace('data_process', 'data')
        event = 'ipo'
        train, test, train_label, test_label = load_data(file_path=os.path.join(dir_path, event), train_file='train_%s.txt' % group, test_file='test_%s.txt' % group,
                  train_label='train_label_%s.txt' % group, test_label='test_label_%s.txt' % group)
        index2word, word2index = build_vocab(train+test, min_df=0)
        label2index, index2label = label2category(train_label+test_label)

        train_data_idx = transform_data(train, word2index)
        label_data_idx = transform_data(train_label, label2index)
        train_idx, val_idx, train_text, val_text, train_labels, val_labels = create_val_test(train_data_idx, train, label_data_idx)

        test_data_idx = transform_data(test, word2index)
        test_label_idx = transform_data(test_label, label2index)
        cPickle.dump([train_idx, train_labels, val_idx, val_labels, test_data_idx, test_label_idx,
                      word2index, index2word, label2index, index2label],
                     open(os.path.join(dir_path, event, "corpus_%s.p" % group), "wb"))

        print "Dataset %s created!" % group
