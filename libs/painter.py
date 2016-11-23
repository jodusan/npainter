from libs import utils
import numpy as np
import tensorflow as tf


def build_model(xs, ys, n_neurons, n_layers, activation_fn, final_activation_fn):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if(activation_fn == 'relu'):
        print("relucina")
        activation_fn = tf.nn.relu
    else:
        activation_fn = tf.nn.tanh

    if(final_activation_fn == 'relu'):
        print("relucina dva")
        final_activation_fn = tf.nn.relu
    else:
        final_activation_fn = tf.nn.tanh

    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')

    n_xs = xs.shape[1]
    n_ys = ys.shape[1]

    X = tf.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=None,
        name='pred')[0]

    cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))

    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}


def train(img,
          gif_step = 100,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=2,
          n_neurons=54,
          n_layers=12,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh):

    H, W, C = img.shape
    xs, ys = utils.split_image_normalized(img)

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.initialize_all_variables())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(
                    it_i + 1, n_iterations, training_cost / n_batches))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            if ((it_i + 1) % gif_step == 0) or (it_i == n_iterations-1):
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess)
                img = ys_pred.reshape(img.shape)
                gifs.append(img)
        return gifs
