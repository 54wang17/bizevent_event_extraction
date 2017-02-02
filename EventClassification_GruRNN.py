from model.event_classification.gru_rnn import GruRNN
import os
import cPickle
import numpy as np
import theano
from model.utils import idx2onehot

_EVENT_TYPE = os.environ.get('EVENT_TYPE', 'Layoff')
_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
_CLASSIFIER_NUM = int(os.environ.get('CLASSIFIER_NUM', '3'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
_BATCH_SIZE = int(os.environ.get('BATCH_SIZE','50'))
CLASSIFIER_NUM = {'IPO': 3, 'Layoff': 2}
ROOT_DIR = {'IPO': 'ipo', 'Layoff': 'layoff'}

event_type = 'IPO'
nepoch = _NEPOCH
batch_size = _BATCH_SIZE
lr = _LEARNING_RATE
in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = CLASSIFIER_NUM[event_type]
root_dir = os.path.dirname(os.path.realpath(__file__))
data_group = 'a'

# load data
data = cPickle.load(open(os.path.join(root_dir, "data", 'event_classification', event_type.lower(), "corpus_%s.p" % data_group), "rb"))
W = cPickle.load(open(os.path.join(root_dir, "data", 'event_classification', event_type.lower(), "word2vec_%s.p" % data_group), "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
train_X, valid_X = data[0], data[2]
train_Y, valid_Y = data[1], data[3]
train_x = [np.matrix(W2V[sen_idx]) for sen_idx in train_X]
train_y = [idx2onehot(label, out_dim) for l in train_Y for label in l]
valid_x = [np.matrix(W2V[sen_idx]) for sen_idx in valid_X]
valid_y = [idx2onehot(label, out_dim) for l in valid_Y for label in l]


model_dir = os.path.join(root_dir, 'model', 'event_classification', 'GRUMiniBatch_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch))
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Initialize and build model
model = GruRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)

# Begin training model with mini batch
model.train_with_mini_batch(data_group, train_x+valid_x, train_y+valid_y, valid_x, valid_y,
                            batch_size=batch_size, learning_rate=lr, nepoch=nepoch,
                            save_path=model_dir)

# If do not want mini batch, use train with sdg instead
# model_dir = os.path.join(root_dir, 'model', 'event_classification', 'GRUSdg_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch))
# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)
# model.train_with_sdg(train_x, train_y, valid_x, valid_y, learning_rate=_LEARNING_RATE, nepoch=_NEPOCH, save_path=model_dir)