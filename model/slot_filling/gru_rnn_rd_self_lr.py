from basic_rnn import BasicRNN
import numpy as np
import theano as theano
import theano.tensor as T
from datetime import datetime
import sys
import os

class GruRNN(BasicRNN):

    def __init__(self, in_dim, hidden_dim, out_dim, word_size, bptt_truncate=-1, activation='tanh'):
        BasicRNN.__init__(self, in_dim, out_dim, hidden_dim, activation)
        # Assign instance variables
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, in_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (out_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(out_dim)
        emb = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_size, in_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.emb = theano.shared(name='emb', value=emb.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.memb = theano.shared(name='emb', value=np.zeros(emb.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.params = [self.U, self.W, self.V, self.b, self.c, self.emb,
                       self.mU, self.mV, self.mW, self.mb, self.mc, self.memb]



    # def calculate_loss(self, x_set, y_set):
    #     return np.mean([self.ce_error(x, y) for x, y in zip(x_set, y_set)])


    def train_with_sdg(self, X_train, y_train, X_valid, y_valid, learning_rate=0.005, nepoch=1, evaluate_loss_after=5, save_path='./data'):
        # We keep track of the losses so we can plot them later
        self.build_model()
        file = open(os.path.join(save_path, 'training_log.txt'), 'w+')
        file.write('Learning Rate:%f\tEpoch Number:%d\t'% (learning_rate, nepoch))
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
                    file.write('%s: num_examples_seen: %d\tepoch: %d\tValidation Loss: %f\n'%(time, num_examples_seen, epoch, loss))
                losses.append((num_examples_seen, loss))
                loss = self.calculate_loss(X_train, y_train)
                print "%s: Training Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                file.write('%s: num_examples_seen: %d\tepoch: %d\tTraining Loss: %f\n'%(time, num_examples_seen, epoch, loss))
                sys.stdout.flush()
                # ADDED! Saving model parameters
                self.save_model_parameters(os.path.join(save_path, "GruSdg"))
        file.close()

    # def get_prediction(self, x_l):
    #     return [self.predict_class(x) for x in x_l]


    def build_model(self):
        V, U, W, b, c, emb = self.V, self.U, self.W, self.b, self.c, self.emb

        idx = T.ivector('idx')
        y = T.ivector('y')
        x = emb[idx]
        def forward_prop_step(x_t, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(s_t_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t) + c)[0]
            return [o_t, s_t]

        [p_y, s], _ = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))])

        # Final output calculation
        prediction = T.argmax(p_y, axis=1)
        o_error = T.mean(T.nnet.categorical_crossentropy(p_y, y))

        # Total cost (could add regularization here)
        self.cost = o_error

        # Gradients
        dU = T.grad(self.cost, U)
        dW = T.grad(self.cost, W)
        db = T.grad(self.cost, b)
        dV = T.grad(self.cost, V)
        dc = T.grad(self.cost, c)
        demb = T.grad(self.cost, emb)
        # Assign functions
        self.predict_possibility = theano.function([idx], p_y)
        self.predict = theano.function([idx], prediction)
        self.ce_error = theano.function([idx, y], self.cost)
        self.bptt = theano.function([idx, y], [dU, dW, db, dV, dc, demb])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        memb = decay * self.memb + (1 - decay) * demb ** 2

        self.f_update = theano.function(
            [idx, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[
                    (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                    (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                    (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                    (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                    (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                    (emb, emb - learning_rate * demb / T.sqrt(memb + 1e-6)),
                    (self.mU, mU),
                    (self.mW, mW),
                    (self.mV, mV),
                    (self.mb, mb),
                    (self.mc, mc),
                    (self.memb, memb)
                    ])
