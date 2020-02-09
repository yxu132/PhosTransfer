import tensorflow as tf

def loss_mean_squared_error(y, y_output):
    return tf.reduce_mean(tf.square(y-y_output))

def mean_absolute_error(y, y_output):
    return tf.reduce_mean(tf.abs(y-y_output))

def loss_cross_entropy(y_, y_output):
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_output,1e-10,1.0)), reduction_indices=[1]))

def accuracy(y_, y_output):
    correct_predictions = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_output, 1))
    acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return acc

def R(y, y_output):
    y_mean, y_var = tf.nn.moments(y, [0])
    y_output_mean, y_output_var = tf.nn.moments(y_output, [0])
    return tf.reduce_mean((y - y_mean) * (y_output - y_output_mean)) / (tf.sqrt(y_var)*tf.sqrt(y_output_var))

def pssp_eval(y_, y_conv):
    return R(y_[:, 0], y_conv[:, 0]), R(y_[:, 1], y_conv[:, 1]), R(y_[:, 2], y_conv[:, 2]), \
           loss_mean_squared_error(y_[:, 0], y_conv[:, 0]), loss_mean_squared_error(y_[:, 1], y_conv[:, 1]), loss_mean_squared_error(y_[:, 2], y_conv[:, 2]), \
           mean_absolute_error(y_[:, 0], y_conv[:, 0]), mean_absolute_error(y_[:, 1], y_conv[:, 1]), mean_absolute_error(y_[:, 2], y_conv[:, 2])
