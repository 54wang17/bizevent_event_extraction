import numpy as np
import operator

def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.in_dim = U.shape[1]
    model.out_dim = V.shape[0]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d in_dim=%d out_dim=%d" %\
          (path, U.shape[0], U.shape[1], V.shape[0])


def idx2onehot(idx, classifier_num):
    one_hot_matrix = np.zeros(classifier_num)
    one_hot_matrix[idx-1] = 1.0
    # one_hot_matrix = np.reshape(one_hot_matrix, (1, classifier_num))
    return one_hot_matrix


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'W', 'V']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        # used for store estimate gradient for each element in matrix
        # calculate the relative error using norm later
        # est_gradients = np.zeros(parameter.shape)
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)

            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient)+np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s" % (pname)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: ",  estimated_gradient
                print "Backpropagation gradient:", backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            print "Gradient check for parameter %s passed." % (pname)
            it.iternext()

        #     est_gradients[ix] = estimated_gradient
        #
        # relative_error = np.linalg.norm(bptt_gradients[pidx] - est_gradients)/\
        #                  np.linalg.norm(bptt_gradients[pidx] + est_gradients)
        # print relative_error
        # if relative_error > error_threshold:
        #     print "Gradient Check ERROR: parameter=%s" % (pname)
        #     # print "+h Loss: %f" % gradplus
        #     # print "-h Loss: %f" % gradminus
        #     # print "Estimated_gradient: %f" % estimated_gradient
        #     # # print "Backpropagation gradient: %f" % backprop_gradient
        #     # print "Relative Error: %f" % relative_error
        #     return
        # print "Gradient check for parameter %s passed." % (pname)