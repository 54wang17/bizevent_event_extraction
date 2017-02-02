from bizevent_event_extraction.model.slot_filling.gru_rnn_minibatch_rd_self_lr import GruRNN
import cPickle
import theano as theano
import numpy as np
import os
from accuracy import conlleval

_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '200'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '50'))
CLASSIFIER_NUM = {'IPO': 20}

event_type = 'IPO'
in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = CLASSIFIER_NUM[event_type]
root_dir = os.path.dirname(os.path.realpath(__file__))
lr = _LEARNING_RATE
nepoch = _NEPOCH
batch_size = _BATCH_SIZE
data_group = 'a'
print 'batch_size', batch_size, 'n_epoch', nepoch, 'learning rate', lr

folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder):
    os.mkdir(folder)

# Load test data
data = cPickle.load(open(os.path.join(root_dir.replace('evaluation', 'data'), event_type.lower(), "corpus_%s.p" % data_group), "rb"))
W = cPickle.load(open(os.path.join(root_dir.replace('evaluation', 'data'), event_type.lower(), "word2vec_%s.p" % data_group), "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
word2idx, idx2word, label2idx, idx2label = data[6], data[7], data[8], data[9]
idx2label = {key-1: val.replace('b-', 'B-').replace('i-', 'I-') for key, val in idx2label.items()}
label2idx = {key: val-1 for key, val in label2idx.items()}
out_dim = len(idx2label)
test_X, test_Y = data[4], data[5]
valid_X, valid_Y = data[2], data[3]
test_Y = [[label-1 for label in l] for l in test_Y]
valid_Y = [[label-1 for label in l] for l in valid_Y]
test_x, valid_x = test_X, valid_X

model_dir = os.path.join(root_dir.replace('evaluation', 'model'), 'GruMiniBatchRdUp_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch))
test_file = open('GruMiniBatchRdUp_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch),'w+')

for root, dirs, files in os.walk(model_dir, topdown=False):
    valid_F1 = {'p': 0.0, 'r': 0.0, 'f1': 0.0}
    test_F1 = {'p': 0.0, 'r': 0.0, 'f1': 0.0}
    for file in files:
        if not file.endswith('.pkl'):
            continue
        model_file = os.path.join(model_dir,file)
        rnn = GruRNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, word_embedding=W2V)
        rnn.load_model_parameters(model_file)
        rnn.build_minibatch(batch_size)

        predictions_test = [map(lambda x: idx2label[x], y) for y in rnn.get_prediction(test_x)]
        groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_Y]
        words_test = [map(lambda x: idx2word[x], w) for w in test_X]

        predictions_valid = [map(lambda x: idx2label[x], y) for y in rnn.get_prediction(valid_x)]
        groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_Y]
        words_valid = [map(lambda x: idx2word[x], w) for w in valid_X]
        # evaluation // compute the accuracy using conlleval.pl
        res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        print res_valid
        test_file.write('Valid: F1:\t%f Precision:\t%f Recall:\t%f, File: %s' %
                        (res_valid['f1'], res_valid['p'], res_valid['r'], file))
        if res_valid['f1'] > valid_F1['f1']:
            valid_F1 = res_valid
            print 'Best valid F1 updated', res_valid
            test_file.write('\tBest valid F1 updated')
        test_file.write('\n')

        print res_test
        test_file.write('Test: F1:\t%f Precision:\t%f Recall:\t%f, File: %s\n' %
                        (res_test['f1'], res_test['p'], res_test['r'], file))
        if res_test['f1'] > test_F1['f1']:
            test_F1 = res_test
            print 'Best test F1 updated', res_test
            test_file.write('\tBest test F1 updated')
        test_file.write('\n')


