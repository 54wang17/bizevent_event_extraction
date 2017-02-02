from basic_rnn import BasicRNN
import numpy as np
import theano as theano
import theano.tensor as T
from datetime import datetime
import sys
import os

class GruRNN(BasicRNN):

    def __init__(self, in_dim, hidden_dim, out_dim, word_embedding, bptt_truncate=-1, activation='tanh'):
        BasicRNN.__init__(self, in_dim, out_dim, hidden_dim, activation)
        # Assign instance variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        # when using mini_batch, should shift dimension
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, in_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, out_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(out_dim)
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.emb = theano.shared(name='emb', value=word_embedding.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.params = [self.U, self.W, self.V, self.b, self.c, self.emb,
                       self.mU, self.mV, self.mW, self.mb, self.mc]

    def build_minibatch(self, batch_size):
        '''
            dimension:  n_steps * batch_size * embed_dim
        :return:
        '''
        V, U, W, b, c, emb = self.V, self.U, self.W, self.b, self.c, self.emb
        self.batch_size = batch_size
        idx = T.ivector('idx')
        y = T.ivector('y')
        m = T.ivector('mask')
        max_len, batch_size = T.iscalar('max_len'), T.iscalar('batch_size')
        x = T.reshape(emb[idx], (batch_size, max_len, self.in_dim)).dimshuffle((1, 0, 2))
        # x = np.zeros((max_len, batch_size, self.in_dim)).astype(theano.config.floatX)
        self.encode = theano.function([idx, max_len, batch_size], x)
        def forward_prop_step(x_t, s_t_prev):
            # GRU Layer
            z_t = T.nnet.hard_sigmoid(T.dot(x_t, U[0]) + T.dot(s_t_prev, W[0]) + b[0])
            r_t = T.nnet.hard_sigmoid(T.dot(x_t, U[1]) + T.dot(s_t_prev, W[1]) + b[1])
            c_t = T.tanh(T.dot(x_t, U[2]) + T.dot((s_t_prev*r_t), W[2]) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev
            y_t = T.nnet.softmax(T.dot(s_t, V) + c)

            return [s_t, y_t]


        [s, y_t], _ = theano.scan(
            forward_prop_step,
            sequences=[x],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros((batch_size, self.hidden_dim))), None])

        # Final output calculation
        # Theano's softmax returns a matrix with one row, we only need the row
        # p_y = T.nnet.softmax(T.dot(s[-1], V) + c)  # [0]

        y_t = y_t.dimshuffle((1,0,2)).reshape((y_t.shape[0]*y_t.shape[1], y_t.shape[2]))
        y_t1 = y_t[np.nonzero(m)]
        p_y = T.argmax(y_t1, axis=1)
        o_error = T.mean(T.nnet.categorical_crossentropy(y_t1, y))

        # Total cost (could add regularization here)
        self.cost = o_error

        # Assign functions
        self.predict = theano.function([idx, m, max_len, batch_size], y_t1)
        self.predict_class = theano.function([idx, m, max_len, batch_size], p_y)
        self.ce_error = theano.function([idx, m, max_len, batch_size, y], self.cost)

        # # Gradients
        dU = T.grad(self.cost, U)
        dW = T.grad(self.cost, W)
        db = T.grad(self.cost, b)
        dV = T.grad(self.cost, V)
        dc = T.grad(self.cost, c)
        demb = T.grad(self.cost, emb)

        self.bptt = theano.function([idx, m, max_len, batch_size, y], [dU, dW, db, dV, dc, demb])

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
            [idx, m, max_len, batch_size, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[
                    (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                    (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                    (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                    (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                    (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                    (emb, emb - learning_rate * demb),
                    (self.mU, mU),
                    (self.mW, mW),
                    (self.mV, mV),
                    (self.mb, mb),
                    (self.mc, mc)
                    ])

    def calculate_loss(self, x_set, y_set, batch_size=1):
        num_train = len(y_set)
        loss = []
        for i in xrange(0, num_train, batch_size):
            # One batch
            s, e = i, min(num_train, i+batch_size)
            X_batch = x_set[s:e]
            y_batch = y_set[s:e]
            idx, x_mask, max_len, batch_size, y = self.prep_batch_data(X_batch, y_batch)
            loss.append(self.ce_error(idx, x_mask, max_len, batch_size, y))
        return np.mean(loss)

    def get_prediction(self, x_set):
        prediction = []
        idx, x_mask, max_len, batch_size = self.prep_batch_data(x_set)
        y_prediction = self.predict_class(idx, x_mask, max_len, batch_size)
        # convert y_prediction to be a batch of predicted labels
        y_i = 0
        for j in xrange(len(x_set)):
            prediction.append(y_prediction[y_i:y_i+len(x_set[j])])
            y_i += len(x_set[j])
        return prediction


    def prep_batch_data(self, x_set, y_set=None):
        lengths = [len(x) for x in x_set]
        max_len, batch_size = max(lengths), len(x_set)
        x_mask = np.zeros((batch_size, max_len), dtype=np.int)
        x = np.zeros((batch_size, max_len), dtype=np.int)
        for idx, s in enumerate(x_set):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.reshape(x_mask, batch_size*max_len).tolist()
        x = np.reshape(x, batch_size*max_len).tolist()
        if y_set is None:
            return x, x_mask, max_len, batch_size
        y = [i for tmp in y_set for i in tmp]
        return x, x_mask, max_len, batch_size, y



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
                sys.stdout.flush()
                # ADDED! Saving model parameters
                self.save_model_parameters(os.path.join(save_path, "GruMiniBatch"))
            # For each training example...
            for i in xrange(0, num_train, batch_size):
                if i+batch_size > num_train:
                    print num_train-i
                s, e = i, min(num_train, i+batch_size)
                X_batch = X_train[s:e]
                y_batch = y_train[s:e]
                idx, x_mask, max_len, batch_size, y = self.prep_batch_data(X_batch, y_batch)
                self.f_update(idx, x_mask, max_len, batch_size, y, learning_rate)
                num_examples_seen += batch_size
                iter_num += 1
        file.close()