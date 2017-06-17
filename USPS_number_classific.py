# Coursera_NN_for_ML
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.axes as axes

# main
def main():
    '''# Initial Conditions
    wd_coefficient = 0.0
    n_hid = 0
    n_iters = 0
    learning_rate = 0.0
    momentum_multiplier = 0.0
    do_early_stopping = False
    mini_batch_size = 0
    # run classifier


    wd_coefficient = 1e-2
    n_hid = 7
    n_iters = 1000
    learning_rate = 0.001
    momentum_multiplier = 0.01
    do_early_stopping = False
    mini_batch_size = 500

    a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier,
       do_early_stopping, mini_batch_size)
    '''
    # Q2
    # a3(0.0,0,0,0.0,0.0,False,0)   # returns error
    # Q3
    a3(0, 10, 70, 0.005, 0, False, 4)

    # Q4
    a3(0, 10, 70, 0.5, 0, False, 4)

def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier,
       do_early_stopping, mini_batch_size):
    model = initial_model(n_hid)
    
    # file format
    mat_data_format = sio.whosmat('data_week9.mat')
    print("MATLAB file format:")
    # import data
    from_data_file = sio.loadmat('data_week9.mat')
    # print(from_data_file)
    datas = from_data_file['data']

    # dataset
    data = datas[0][0]
    # Training set
    train_inputs = data['training']['inputs'][0][0]
    train_targets = data['training']['targets'][0][0]
    #print(train_targets)
    # Validation set
    val_inputs = data['validation']['inputs'][0][0]
    val_targets = data['validation']['targets'][0][0]
    
    # Test set
    test_inputs = data['test']['inputs'][0][0]
    test_targets = data['test']['targets'][0][0]
    
    size_train = train_inputs.shape
    n_training_cases = size_train[1]

    # dataset
    data = datas[0][0]
    # Training set
    train = {}
    train['inputs'] = data['training']['inputs'][0][0]
    train['targets'] = data['training']['targets'][0][0]
    # print(train_targets)

    # Validation set
    valid = {}
    valid['inputs'] = data['validation']['inputs'][0][0]
    valid['targets'] = data['validation']['targets'][0][0]

    # Test set
    test = {}
    test['inputs'] = data['test']['inputs'][0][0]
    test['targets'] = data['test']['targets'][0][0]

    n_training_cases = train['inputs'].shape[1]

    if n_iters != 0:
        test_gradient(model, train, wd_coefficient)

    ## Optimization
    theta = model_to_theta(model)
    momentum_speed = theta * 0.0


    training_data_losses = []
    validation_data_losses = []


    if do_early_stopping == True:
        best_so_far = {}
        best_so_far['theta'] = -1.0;  # this will be overwritten soon
        best_so_far['validation_loss'] = np.inf
        best_so_far['after_n_iters'] = -1.0

    # Iterate through optimizations
    for i in range(0, n_iters):
        model = theta_to_model(theta);

        training_batch_start = ((i - 1) * mini_batch_size % n_training_cases) + 1
        training_batch = {}
        training_batch['inputs'] = train['inputs'][:, training_batch_start: training_batch_start + mini_batch_size - 1]
        training_batch['targets'] = train['targets'][:,
                                    training_batch_start: training_batch_start + mini_batch_size - 1]

        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient))
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        theta = theta + momentum_speed * learning_rate

        model = theta_to_model(theta)
        # print(training_data_losses)
        # print(loss(model, train, wd_coefficient))
        '''
        training_data_losses = np.concatenate((training_data_losses,loss(model, train, wd_coefficient)))
        validation_data_losses = np.concatenate((validation_data_losses, loss(model, valid, wd_coefficient)))
        '''
        training_data_losses.append(loss(model, train, wd_coefficient))
        validation_data_losses.append(loss(model, valid, wd_coefficient))

        # print(validation_data_losses[-1])
        if do_early_stopping == True:
            if validation_data_losses[-1] < best_so_far['validation_loss']:
                best_so_far['theta'] = theta
                best_so_far['validation_loss'] = validation_data_losses[-1]
                best_so_far['after_n_iters'] = i

                if (i % round(n_iters / 10)) == 0:
                    print('After %d optimization iterations, training data loss is %f, and validation data loss is %f\n',
                          i, training_data_losses(-1), validation_data_losses(-1))

    if n_iters != 0:
        test_gradient(model, train, wd_coefficient)

    if do_early_stopping == True:
        print('Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.\n',
              best_so_far['after_n_iters'])
        theta = best_so_far['theta']

    # the optimization is finished. Now do some reporting.
    model = theta_to_model(theta)

    if n_iters != 0:

        x=np.arange(1,n_iters+1)
        fig1 = plt.subplots()
        plt.plot(x, training_data_losses, c='b', label='training')
        plt.plot(x, validation_data_losses, c='r', label='validation')

        # Now add the legend with some customizations.
        # legend = ax.legend(loc='upper center', shadow=True)

        plt.xlim(0, n_iters)
        #plt.ylim(0, 1)
        plt.xlabel('Number iterations')
        plt.title('Validation/Training Loss')

        # ax.legend('training', 'validation')
        # ax.ylabel('loss')
        # ax.xlabel('iteration number')
        plt.show()

    datas2 = (train, valid, test)
    data_names = ('training', 'validation', 'test')
    for i in range(0,len(datas2)):
        data = datas2[i]
        data_name = data_names[i]
        print('\nThe loss on the %s data is %f\n', data_name, loss(model, data, wd_coefficient))
        if wd_coefficient != 0:
            print('The classification loss (i.e. without weight decay) on the %s data is %f\n',
                  data_name, loss(model, data, 0))

        print('The classification error rate on the %s data is %f\n',
              data_name, classification_performance(model, data))


    


def test_gradient(model, data, wd_coefficient):
    base_theta = model_to_theta(model)
    h = 1e-2;
    correctness_threshold = 1e-5;
    analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));
    # Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
    for i in range(0,100):
        test_index = (i * 1299721 % base_theta.shape[0]) + 1     # 1299721 is prime and thus ensures a somewhat random-like selection of indices
        analytic_here = analytic_gradient[test_index]
        theta_step = base_theta * 0
        theta_step[test_index] = h
        contribution_distances = np.concatenate((np.arange(-4,0), np.arange(1,5)))
        # print(contribution_distances)
        contribution_weights = [1/280.0, -4/105.0, 1/5.0, -4/5.0, 4/5.0, -1/5.0, 4/105.0, -1/280.0]
        temp = 0
        for contribution_index in range(0,8):
            temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances[contribution_index]),
                             data, wd_coefficient) * contribution_weights[contribution_index]

        fd_here = temp / h
        diff = abs(analytic_here - fd_here)
        # print('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
        if diff < correctness_threshold:
            pass
        if diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold:
            pass
        print('Theta element #%d, with value %e, has finite difference gradient %e but analytic gradient %e. ' +
              'That looks like an error.\n', test_index, base_theta[test_index], fd_here, analytic_here)

    print('Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the ' +
    'gradient that the finite difference approximation computed, so the gradient calculation procedure is probably '+
    'correct (not certainly, but probably).\n')

def logistic(input):
    return 1.0 / (1.0 + np.exp(-input))


def log_sum_exp_over_rows(a):
  # This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = np.amax(a, axis=0)
  # print(maxs_small.shape)
  # print(a.shape[0])
  maxs_big = np.matlib.repmat(maxs_small, a.shape[0], 1)

  return np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small

def loss(model, data, wd_coefficient):
    # model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
    # model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
    # data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    # data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

    # Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.
    # print(model['input_to_hid'].shape)
    # print(data['inputs'].shape)
    hid_input = np.matmul(model['input_to_hid'], data['inputs'])      # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)                        # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    # print(model['hid_to_class'].shape)
    # print(hid_output.shape)
    class_input = np.matmul(model['hid_to_class'], hid_output)           # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>

    # The following three lines of code implement the softmax.
    # However, it's written differently from what the lectures say.
    # In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    # What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable.
    # "Numerically stable" means that this way, there will never be really big numbers involved.
    # The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
    # Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
    class_normalizer = log_sum_exp_over_rows(class_input) # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.matlib.repmat(class_normalizer, class_input.shape[0], 1)     # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    class_prob = np.exp(log_class_prob)     # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>


    # print(log_class_prob.shape)
    # print(data['targets'].shape)
    classification_loss = -np.mean(np.sum(np.multiply(log_class_prob,data['targets']), 0))  # select the right log class probability using that sum; then take the mean over all data cases.
    wd_loss = np.sum(model_to_theta(model)**2)/2*wd_coefficient # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    return classification_loss + wd_loss

def d_loss_by_d_model(model, data, wd_coefficient):
    # model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
    # model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    # data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    # data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

    # The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.

    # This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to replace that by a correct computation.
    n_data = data['inputs'].shape[1]

    # FORWARD PASS
    # activation for 1st layer
    x1 = data['inputs']

    # Input for 2nd layer
    s2 = np.matmul(model['input_to_hid'], x1)

    # Activation for 2nd layer
    x2 = logistic(s2)

    # Input for 3rd layer (Softmax)
    s3 = np.matmul(model['hid_to_class'], x2)

    # Output for 3rd layer (Softmax)
    # x3 = np.divide(np.exp(s3),np.sum(np.exp(s3),axis=0))  # this is numerically unstable
    class_normalizer = log_sum_exp_over_rows(s3)    # log(sum(exp of class_input)) is what we subtract to get
        # properly normalized log class probabilities.size: <1 > by < number of data cases >

    log_class_prob = s3 - np.matlib.repmat(class_normalizer, s3.shape[0], 1) # log of probability of each
        # class . size: <number of classes, i.e. 10 > by < number of data cases >
    x3 = np.exp(log_class_prob)

    # HIDDEN TO CLASS
    dE_ds3 = np.subtract(x3,data['targets'])    # 10 x 1000
    # print("dE_ds3")
    # print(dE_ds3.shape)
    ds3_dw23 = x2
    # print("ds3_dw23")
    # print(ds3_dw23.shape)
    dE_dw23 = np.transpose(np.matmul(ds3_dw23,np.transpose(dE_ds3)))
    # print("dE_dw23")
    # print(dE_dw23.shape)

    # INPUT TO HIDDEN
    dw23_ds2 = model['hid_to_class']    # w23 (7x10)
    # print("dw23_ds2")
    # print(dw23_ds2.shape)
    ds2_dw12 = np.subtract(np.ones(x2.shape),x2) # 7 x 1000
    # print("ds2_dw12")
    # print(ds2_dw12.shape)

    x2_1_x2 = np.multiply(ds2_dw12,x2)   # 7 x 1000
    # print("x2_1_x2")
    # print(x2_1_x2.shape)
    # (xi-ti)wji
    dE_ds2 = np.matmul(np.transpose(dw23_ds2),dE_ds3)    # 7 x 1000
    # print("dE_ds2")
    # print(dE_ds2.shape)
    x3_t3_w23 = np.multiply(x2_1_x2, dE_ds2)    # 7 x 1000
    # print("x3_t3_w23")
    # print(x3_t3_w23.shape)

    ds2_dw12 = data['inputs']       # (256 x 1000)

    dE_dw12 = np.transpose(np.matmul(ds2_dw12,np.transpose(x3_t3_w23)))      # (256 x 7)
    # print("dE_dw12")
    # print(dE_dw12.shape)

    # final values with regularization term (weight decay)
    dE_dw12 = np.add(dE_dw12, wd_coefficient * model['input_to_hid'])
    dE_dw23 = np.add(dE_dw23, wd_coefficient * model['hid_to_class'])
    ret = {}
    ret['input_to_hid'] = np.add(dE_dw12,wd_coefficient * model['input_to_hid'])/n_data
    ret['hid_to_class'] = np.add(dE_dw23,wd_coefficient * model['hid_to_class'])/n_data
    return ret

def model_to_theta(model):
  # This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
  input_to_hid_transpose = np.transpose(model['input_to_hid'])
  # print(input_to_hid_transpose.shape)
  hid_to_class_transpose = np.transpose(model['hid_to_class'])
  # print(hid_to_class_transpose.shape)
  return np.concatenate((input_to_hid_transpose.flatten(),hid_to_class_transpose.flatten()))

def theta_to_model(theta):
    # This function takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    n_hid = int(theta.shape[0] / (256+10))
    ret = {}
    # print(theta.shape)

    # print(np.reshape(theta[0 : 256*n_hid], [256, n_hid]))
    ret['input_to_hid'] = np.transpose(np.reshape(theta[0 : 256*n_hid], [256, n_hid]))
    ret['hid_to_class'] = np.transpose(np.reshape(theta[256 * n_hid : theta.shape[0]], (n_hid, 10)))
    return ret

def initial_model(n_hid):
    n_params = (256+10) * n_hid
    as_row_vector = np.cos(np.arange(0,n_params))
    return theta_to_model(np.vstack(as_row_vector) * 0.1) # We don't use random initialization, for this assignment. This way, everybody will get the same results.


def classification_performance(model, data):
    # This returns the fraction of data cases that is incorrectly classified by the model.
    hid_input = np.matmul(model['input_to_hid'], data['inputs'])    # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)      # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = np.matmul(model['hid_to_class'],hid_output) # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>

    choices = np.argmax(class_input, axis=0)    # choices is integer: the chosen class, plus 1.
    targets = np.argmax(data['targets'], axis=0)       # targets is integer: the target class, plus 1.

    return np.sum(choices != targets)/(targets.shape[0])

# run Main
main()
