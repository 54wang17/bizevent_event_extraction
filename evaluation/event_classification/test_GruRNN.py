from bizevent_event_extraction.model.event_classification.gru_rnn import GruRNN
import cPickle
import theano as theano
import numpy as np
import os
from sklearn.metrics import confusion_matrix

_WORD_EMBEDDING_SIZE = int(os.environ.get('Word_2_Vec_DIM', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
CLASSIFIER_NUM = {'IPO': 3, 'Layoff': 2}
CLASSIFIER_LABELS = {'IPO': ['N/A', 'Upcoming', 'Priced'], 'Layoff': ['Layoff', 'N/A']}
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))
_BATCH_SIZE = int(os.environ.get('BATCH_SIZE','50'))

batch_size = _BATCH_SIZE
lr = _LEARNING_RATE
nepoch = _NEPOCH
event_type = 'Layoff'
data_group = 'a'
in_dim = _WORD_EMBEDDING_SIZE
hidden_dim = _HIDDEN_DIM
out_dim = CLASSIFIER_NUM[event_type]
root_dir = os.path.dirname(os.path.realpath(__file__))

labels = CLASSIFIER_LABELS[event_type]
labels.append('Total')


# Load test data
data = cPickle.load(open(os.path.join(root_dir.replace('evaluation', 'data'), event_type.lower(), "corpus_%s.p" % data_group), "rb"))
W = cPickle.load(open(os.path.join(root_dir.replace('evaluation', 'data'), event_type.lower(), "word2vec_%s.p" % data_group), "rb"))
W2V = np.array(W[0]).astype(theano.config.floatX)
test_X, test_y = data[4], data[5]
test_X = [np.matrix(W2V[sen_idx]) for sen_idx in test_X]
model_dir = os.path.join(root_dir.replace('evaluation', 'model'), 'GRUMiniBatch_%s_%f_%d_%d' % (data_group, lr, batch_size, nepoch))

test_file = open(os.path.join(root_dir, '%d_%s_gru_minibatch_test_%f_%d_50.txt' %
                              (hidden_dim, data_group, lr, batch_size)), 'w+')
for root, dirs, files in os.walk(model_dir, topdown=False):
    count = 0
    for file in files:
        if not file.endswith('.pkl'):
            continue
        model_file = os.path.join(model_dir, file)
        rnn = GruRNN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        rnn.load_model_parameters(model_file)

        rnn.build_minibatch(batch_size=50)
        predict_y = rnn.get_prediction(test_X, minibatch=True)

        # Generate Evaluation Matrix
        test_y_ = [y for l in test_y for y in l]
        predict_y_ = [y+1 for x in predict_y for y in x.tolist()][:len(test_y)]
        eval_matrix = confusion_matrix(test_y_, predict_y_)

        if len(eval_matrix) != len(labels):
            eval_matrix = eval_matrix
            eval_matrix = np.insert(eval_matrix, len(labels)-1, 0, axis=1)
            eval_matrix = np.vstack([[0.0]*len(labels), eval_matrix])

        # use the first row and last column to sum total number of instances for each category
        eval_matrix = np.asmatrix(eval_matrix, dtype=int)
        for i in xrange(1, eval_matrix.shape[0]):
            eval_matrix[i, -1] = np.sum(eval_matrix[i])
            eval_matrix[0, i-1] = np.sum(eval_matrix[:, i-1])

        # Pretty print the evaluation matrix
        print '\t\t\t\t\tPredicted\n\t\t\t'+'\t'.join(labels)
        for i in xrange(eval_matrix.shape[0]):
            print '{:10s}'.format(labels[i-1]),
            for j in xrange(eval_matrix.shape[1]):
                print '{:5d}'.format(eval_matrix[i, j]),
            print
        precision = []
        recall = []
        f_score = []
        e = 0
        for i in xrange(1, eval_matrix.shape[0]):
            r = 1.0*eval_matrix[i,i-1]/eval_matrix[i, -1]
            if eval_matrix[i, i-1] == 0:
                p = 0.0
                e += 1
            else:
                p = 1.0*eval_matrix[i, i-1]/eval_matrix[0, i-1]
            if abs(p+r-0.0) <1e-05:
                f = 0.0
            else:
                f = 2*p*r/(p+r)
            precision.append(p)
            recall.append(r)
            f_score.append(f)

        if e and e != len(precision):
            precision.append(np.sum(precision)/(len(precision)-e))
        else:
            precision.append(np.mean(precision))
        recall.append(np.mean(recall))
        f_score.append(np.mean(f_score))
        print 'P:', precision[-1], 'R:', recall[-1], 'F1', f_score[-1]

        for p in precision:
            test_file.write(str(p)+',\t')
        for r in recall:
            test_file.write(str(r)+',\t')
        for f in f_score:
            test_file.write(str(f)+',\t')
        test_file.write(file+'\n')

        print '-'*50
