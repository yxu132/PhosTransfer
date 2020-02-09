import tensorflow as tf
import evaluation

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

class PhosphoNet:

    def __init__(self, VEC_SIZE=23, WINDOW_SIZE=9, OUTPUT_SIZE=2, LR_ALPHA=1e-4,
                 hidden_layers=3, is_train=True,
                 is_context=False, is_weighted=False,
                 add_hidden_layer=True):

        self._hidden_layers = hidden_layers
        self._fixed_hidden_layers = hidden_layers -1

        input_size = VEC_SIZE*(WINDOW_SIZE*2+1)

        self._input = tf.placeholder(tf.float32, shape=[None, input_size])
        self._label = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
        self._context = tf.placeholder(tf.float32, shape=[None, 100])

        raw = tf.reshape(self._input, shape=[-1, 1, (WINDOW_SIZE * 2 + 1), VEC_SIZE])

        with tf.variable_scope('hidden_layer_1'):
            w_initial = tf.random_normal_initializer(stddev=0.1)
            b_initial = tf.constant_initializer(0.1)

            # First layer: convolution
            W_conv1 = tf.get_variable("W_conv1", [1, 5, VEC_SIZE, VEC_SIZE*2],
                                      initializer=w_initial)  # input feature map size = 1, output feature map size = 32
            b_conv1 = tf.get_variable("b_conv1", [VEC_SIZE*2], initializer=b_initial)
            h_conv1 = tf.nn.relu(conv2d(raw, W_conv1) + b_conv1)  # convolution layer with window 5x5, strides len = 1, padding = 'SAME'
            h_pool1 = max_pool_2x2(h_conv1)  # max_pooing with window = 2x2

            if hidden_layers == 1:
                self.top_layers(OUTPUT_SIZE, w_initial, b_initial, h_pool1, is_context, is_train, is_weighted, LR_ALPHA)

        # Second layer: convolution
        with tf.variable_scope('hidden_layer_2'):
            W_conv2 = tf.get_variable("W_conv2", [1, 5, VEC_SIZE*2, VEC_SIZE*4], initializer=w_initial)
            b_conv2 = tf.get_variable("b_conv2", [VEC_SIZE*4], initializer=b_initial)
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            if hidden_layers == 2:
                self.top_layers(OUTPUT_SIZE, w_initial, b_initial, h_pool2, is_context, is_train, is_weighted, LR_ALPHA)

        # Third layer: convolution
        with tf.variable_scope('hidden_layer_3'):
            W_conv3 = tf.get_variable("W_conv3", [1, 3, VEC_SIZE*4, VEC_SIZE*8], initializer=w_initial)
            b_conv3 = tf.get_variable("b_conv3", [VEC_SIZE*8], initializer=b_initial)
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

            if hidden_layers == 3:
                self.top_layers(OUTPUT_SIZE, w_initial, b_initial, h_pool3, is_context, is_train, is_weighted, LR_ALPHA)

        # Third layer: convolution
        if hidden_layers >= 4:

            with tf.variable_scope('hidden_layer_4'):
                W_conv4 = tf.get_variable("W_conv4", [1, 3, VEC_SIZE * 8, VEC_SIZE * 16], initializer=w_initial)
                b_conv4 = tf.get_variable("b_conv4", [VEC_SIZE * 16], initializer=b_initial)
                h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
                # h_pool4 = h_conv4
                h_pool4 = max_pool_2x2(h_conv4)

                if hidden_layers == 4:
                    self.top_layers(OUTPUT_SIZE, w_initial, b_initial, h_pool4, is_context, is_train, is_weighted, LR_ALPHA)


                # Third layer: convolution
        if hidden_layers >= 5:
            with tf.variable_scope('hidden_layer_5'):
                W_conv5 = tf.get_variable("W_conv5", [1, 3, VEC_SIZE * 16, VEC_SIZE * 32], initializer=w_initial)
                b_conv5 = tf.get_variable("b_conv5", [VEC_SIZE * 32], initializer=b_initial)
                h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
                # h_pool5 = h_conv5
                h_pool5 = max_pool_2x2(h_conv5)

                if hidden_layers == 5:
                    self.top_layers(OUTPUT_SIZE, w_initial, b_initial, h_pool5, is_context, is_train, is_weighted, LR_ALPHA)


    def top_layers(self, OUTPUT_SIZE, w_initial, b_initial, h_pool,
                   is_context, is_train, is_weighted, LR_ALPHA):

        pool_size = tf.size(h_pool)
        if self._hidden_layers == 5:
            pool_size = 832
        elif self._hidden_layers == 4:
            pool_size = 832
        elif self._hidden_layers == 3:
            pool_size = 624
        elif self._hidden_layers == 2:
            pool_size = 520
        elif self._hidden_layers == 1:
            pool_size = 520
        # Fully connected layer
        W_fc1 = tf.get_variable("W_fc1", [pool_size, 512], initializer=w_initial)
        b_fc1 = tf.get_variable("b_fc1", [512], initializer=b_initial)
        h_pool_flat = tf.reshape(h_pool, [-1, pool_size])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        # Dropout
        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        W_fc2 = tf.get_variable("W_fc2", [512, OUTPUT_SIZE], initializer=w_initial)
        b_fc2 = tf.get_variable("b_fc2", [OUTPUT_SIZE], initializer=b_initial)

        if is_context:
            W_ctxt = tf.get_variable('W_ctxt', [100, OUTPUT_SIZE], initializer=w_initial)
            logits = tf.matmul(h_fc1_drop, W_fc2) + tf.matmul(self._context, W_ctxt) + b_fc2
            self._output = tf.nn.softmax(logits)
        else:
            logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            self._output = tf.nn.softmax(logits)

        if not is_train:
            return

        # self._loss = loss_function(self._label, self._output)
        self._weights = tf.placeholder(tf.float32, shape=[None])
        if is_weighted:
            self._loss = tf.losses.softmax_cross_entropy(self._label, logits, weights=self._weights)
        else:
            self._loss = tf.losses.softmax_cross_entropy(self._label, logits)

        # Train_op
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "hidden_layer_"+str(self._hidden_layers))
        self._train_step = tf.train.AdamOptimizer(LR_ALPHA).minimize(self._loss, var_list=train_vars)



    @property
    def weights(self):
        return self._weights

    @property
    def input(self):
        return self._input

    @property
    def label(self):
        return self._label

    @property
    def output(self):
        return self._output

    @property
    def train_step(self):
        return self._train_step

    @property
    def loss(self):
        return self._loss

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def context(self):
        return self._context

if __name__ == '__main__':
    model = PhosphoNet(VEC_SIZE=26, WINDOW_SIZE=9, hidden_layers=4, is_context=True)


