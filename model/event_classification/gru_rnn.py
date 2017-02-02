from basic_rnn import BasicRNN
import numpy as np
import theano as theano
import theano.tensor as T
from datetime import datetime
import sys
import os

class GruRNN(BasicRNN):

    def __init__(self, in_dim, hidden_dim, out_dim, bptt_truncate=-1, activation='tanh'):
        BasicRNN.__init__(self, in_dim, out_dim, hidden_dim, activation)
        # Assign instance variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, in_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, out_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(out_dim)
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.params = [self.U, self.W, self.V, self.b, self.c,
                       self.mU, self.mV, self.mW, self.mb, self.mc]

    def build_minibatch(self, batch_size):
        '''
            dimension:  n_steps * batch_size * embed_dim
        :return:
        '''
        V, U, W, b, c = self.V, self.U, self.W, self.b, self.c

        x = T.tensor3('x')
        y = T.matrix('y')
        m = T.matrix('mask')
        self.batch_size = batch_size

        def forward_prop_step(x_t, m_t, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # GRU Layer
            z_t = T.nnet.hard_sigmoid(T.dot(x_t, U[0]) + T.dot(s_t_prev, W[0]) + b[0])
            r_t = T.nnet.hard_sigmoid(T.dot(x_t, U[1]) + T.dot(s_t_prev, W[1]) + b[1])
            c_t = T.tanh(T.dot(x_t, U[2]) + T.dot((s_t_prev*r_t), W[2]) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev
            s_t = m_t[:, None] * s_t + (1.0 - m_t)[:, None] * s_t_prev
            return s_t


        s, _ = theano.scan(
            forward_prop_step,
            sequences=[x, m],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros((batch_size, self.hidden_dim)))])

        # Final output calculation
        # Theano's softmax returns a matrix with one row, we only need the row
        p_y = T.nnet.softmax(T.dot(s[-1], V) + c)  # [0]
        prediction = T.argmax(p_y, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y))/self.batch_size

        # Total cost (could add regularization here)
        self.cost = o_error

        # Assign functions
        self.predict = theano.function([x, m], p_y)
        self.predict_class = theano.function([x, m], prediction)
        self.ce_error = theano.function([x, y, m], self.cost)

        # Gradients
        dU = T.grad(self.cost, U)
        dW = T.grad(self.cost, W)
        db = T.grad(self.cost, b)
        dV = T.grad(self.cost, V)
        dc = T.grad(self.cost, c)

        self.bptt = theano.function([x, y, m], [dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.f_update = theano.function(
            [x, y, m, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

    def train_with_mini_batch(self, group, X_train, y_train, X_valid, y_valid, batch_size=50,
                              learning_rate=0.005, nepoch=1, evaluate_loss_after=1, save_path='./data'):
        self.build_minibatch(batch_size)
        file = open(os.path.join(save_path, 'training_log.txt'), 'w+')
        file.write('Learning Rate:%f\tEpoch Number:%d\tBatch Size:%d\n'% (learning_rate, nepoch, batch_size))
        isValidation = False
        if X_valid is not None:
            assert(y_valid is not None)
            isValidation = True
        num_train = len(y_train)
        losses = []
        num_examples_seen = 0
        iter_num = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                if isValidation:
                    loss = self.calculate_loss(X_valid, y_valid, batch_size)
                    print "%s: Validation Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                    file.write('%s: num_examples_seen: %d\tepoch: %d\tValidation Loss: %f\n'%(time, num_examples_seen, epoch, loss))
                losses.append((num_examples_seen, loss))
                loss = self.calculate_loss(X_train, y_train, batch_size)
                print "%s: Training Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                file.write('%s: num_examples_seen: %d\tepoch: %d\tTraining Loss: %f\n'%(time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                # if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                #     learning_rate *= 0.5
                #     print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
                # ADDED! Saving model parameters
                self.save_model_parameters(os.path.join(save_path, "GruMiniBatch"))
            # For each training example...
            for i in xrange(0, num_train, batch_size):
                # One SGD step
                s, e = i, min(num_train, i+batch_size)
                X_batch = X_train[s:e]
                y_batch = y_train[s:e]
                x, y, x_mask = self.prep_batch_data(X_batch, y_batch, batch_size)
                if e - s < batch_size:
                    self.batch_size = e - s
                self.f_update(x, y, x_mask, learning_rate)
                num_examples_seen += self.batch_size
                iter_num += 1
                # print num_examples_seen,
            # print
        file.close()

    def calculate_loss(self, x_set, y_set, batch_size=1):
        if batch_size == 1:
            return super(BasicRNN, self).calculate_loss(x_set, y_set)
        else:
            num_train = len(y_set)
            loss = []
            for i in xrange(0, num_train, batch_size):
                # One SGD step
                s, e = i, min(num_train, i+batch_size)
                X_batch = x_set[s:e]
                y_batch = y_set[s:e]
                x, y, x_mask = self.prep_batch_data(X_batch, y_batch, batch_size)
                loss.append(self.ce_error(x, y, x_mask))
            return np.mean(loss)

    def train_with_sgd(self, X_train, y_train, X_valid, y_valid, learning_rate=0.005, nepoch=1, evaluate_loss_after=5, save_path='./data'):
        # We keep track of the losses so we can plot them later
        self.build_model()
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
                '''
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate *= 0.5
                    print "Setting learning rate to %f" % learning_rate
                '''
                sys.stdout.flush()
                # ADDED! Saving model parameters
                self.save_model_parameters(os.path.join(save_path, "GruSdg"))

    def get_prediction(self, x_l, minibatch=False):
        if not minibatch:
            return super(BasicRNN, self).get_prediction(x_l)
        num_x = len(x_l)
        prediction = []
        for i in xrange(0, num_x, self.batch_size):
            s, e = i, min(num_x, i + self.batch_size)
            X_batch = x_l[s:e]
            x, x_mask = self.prep_batch_data(X_batch, [], self.batch_size)
            prediction.append(self.predict_class(x, x_mask))
        return prediction[:num_x]

    def build_model(self):
        V, U, W, b, c = self.V, self.U, self.W, self.b, self.c

        x = T.matrix('x')
        y = T.matrix('y')

        def forward_prop_step(x_t, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(s_t_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev

            return s_t

        s, _ = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))])

        # Final output calculation
        # Theano's softmax returns a matrix with one row, we only need the row
        p_y = T.nnet.softmax(V.dot(s[-1]) + c)  # [0]
        prediction = T.argmax(p_y, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(p_y, y))

        # Total cost (could add regularization here)
        self.cost = o_error

        # Gradients
        dU = T.grad(self.cost, U)
        dW = T.grad(self.cost, W)
        db = T.grad(self.cost, b)
        dV = T.grad(self.cost, V)
        dc = T.grad(self.cost, c)

        # Assign functions
        self.predict = theano.function([x], p_y)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], self.cost)
        self.bptt = theano.function([x, y], [dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.f_update = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

    def prep_batch_data(self, x_set, y_set, batch_size):
        lengths = [x.shape[0] for x in x_set]
        max_len = max(lengths)
        x_mask = np.zeros((max_len, batch_size)).astype(theano.config.floatX)
        x = np.zeros((max_len, batch_size, self.in_dim)).astype(theano.config.floatX)
        for idx, s in enumerate(x_set):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.
        if len(y_set) == 0:
            return x, x_mask
        padding_y = []
        if len(y_set) < batch_size:
            padding_y = [np.zeros(y_set[0].shape).astype(theano.config.floatX)]*(batch_size-len(y_set))
        y = np.array(y_set + padding_y)
        return x, y, x_mask
