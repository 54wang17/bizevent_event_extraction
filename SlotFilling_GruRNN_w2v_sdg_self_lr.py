from model.slot_filling.gru_rnn_w2v_self_lr import GruRNN
import os
import cPickle
import numpy as np
import theano

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


data = cPickle.load(open(os.path.join(root_dir, "data", 'slot_filling', event_type.lower(), "corpus_%s.p" % data_group), "rb"))
W = cPickle.load(open(os.path.join(root_dir, "data", 'slot_filling', event_type.lower(), "word2vec_%s.p" % data_group), "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
word2idx, idx2word, label2idx, idx2label = data[6], data[7], data[8], data[9]
train_X, valid_X = data[0], data[2]
train_Y, valid_Y = data[1], data[3]
train_Y = [[label-1 for label in l] for l in train_Y]
valid_Y = [[label-1 for label in l] for l in valid_Y]
train_x, valid_x = train_X, valid_X
out_dim = len(idx2label)

model_dir = os.path.join(root_dir, 'model', 'slot_filling', 'GruSdgW2VUp_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch))
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# initialize model
model = GruRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, word_emb=W2V)
# train with sdg
model.train_with_sdg(train_x, train_Y, valid_x, valid_Y,
            learning_rate=_LEARNING_RATE, nepoch=_NEPOCH, evaluate_loss_after=1, save_path=model_dir)

