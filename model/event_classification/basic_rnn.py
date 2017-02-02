import theano as theano
import theano.tensor as T
import numpy as np
from datetime import datetime
import sys
import cPickle

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'


class BasicRNN(object):

    def __init__(self, in_dim, out_dim, hidden_dim=100, bptt_truncate=4, activation='tanh'):
        # Assign instance variables
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./in_dim), np.sqrt(1./in_dim), (hidden_dim, in_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (out_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.params = [self.U, self.W, self.V]
        if activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0)
        elif activation == 'cappedrelu':
            self.activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError
        # self.build_model()

    def build_model(self):
        U, W, V = self.U, self.W, self.V
        x = T.matrix('x')
        y = T.matrix('y')

        def forward_prop_step(x_t, s_tm1, U, W):
            s_t = self.activation(T.dot(U, x_t) + T.dot(W, s_tm1))
            return s_t

        s, _ = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, W],
            truncate_gradient=self.bptt_truncate,
            mode='DebugMode')

        p_y = T.nnet.softmax(T.dot(V, s[-1]))   # a vector represent the probability of each label
        prediction = T.argmax(p_y, axis=1)      # an index of predicted label
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y))
        self.cost = o_error

        # compute the gradient of cost with respect to theta = (U, W, V)
        # gradients on the weights using BPTT


        # Assign functions
        self.forward_propagation = theano.function([x], s[-1])
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)


        learning_rate = T.scalar('learning_rate')
        self.bptt, self.f_update = self.SDG(x, y, learning_rate)

    def calculate_loss(self, x_set, y_set):
        return np.mean([self.ce_error(x, y) for x, y in zip(x_set, y_set)])

    def get_prediction(self, x_l):
        return [self.predict(x) for x in x_l]

    def train_with_sgd(self, X_train, y_train, X_valid, y_valid, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later

        isValidation = False
        if X_valid is not None:
            assert(y_valid is not None)
            isValidation = True
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.f_update(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                if isValidation:
                    loss = self.calculate_loss(X_valid, y_valid)
                    print "%s: Validation Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                losses.append((num_examples_seen, loss))
                loss = self.calculate_loss(X_train, y_train)
                print "%s: Training Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate *= 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
                # ADDED! Saving model parameters
                self.save_model_parameters("Basic")

    def SDG(self, x, y, lr):
        # SGD
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)
        bptt = theano.function([x, y], gparams)
        updates = []
        for param, gparam in zip(self.params, gparams):
            upd = param - lr * gparam
            updates.append((param, upd))
        sgd_step = theano.function([x, y, lr], [], updates=updates)
        return bptt, sgd_step

    def save_model_parameters(self, prefix):
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        file_name = prefix + "rnn-w2v-%d-%d-%s.pkl" % (self.hidden_dim, self.in_dim, time)
        with open(file_name, "wb") as model_file:
            params_vale = [p.get_value() for p in self.params]
            cPickle.dump(params_vale, model_file, protocol=cPickle.HIGHEST_PROTOCOL)
            print "Saved model parameters to %s." % model_file

    def load_model_parameters(self, path_to_file):
        data = cPickle.load(open(path_to_file, "rb"))
        i = iter(data)
        for param in self.params:
            param.set_value(i.next())
