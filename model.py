import tensorflow as tf

tf.app.flags.DEFINE_string('model', 'lstm', 'RNN cell type, selected from "rnn", "gru" and "lstm"')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Size of a training batch')
tf.app.flags.DEFINE_integer('num_steps', 50, 'Number of steps to unroll the RNN')
tf.app.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden layer size in the RNN')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of cell layers')
tf.app.flags.DEFINE_integer('vocab_size', 65, 'Vocabulary size')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'The maximum norm used to clip the gradients')

FLAGS = tf.app.flags.FLAGS

def build_cell(keep_prob):
    dim = FLAGS.hidden_size
    if FLAGS.model == 'rnn':
        cell_unit = tf.nn.rnn_cell.BasicRNNCell(num_units=dim)
    elif FLAGS.model == 'gru':
        cell_unit = tf.nn.rnn_cell.GRUCell(num_units=dim)
    elif FLAGS.model == 'lstm':
        cell_unit = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim)
    else:
        raise Exception('Model not found. Must be "rnn", "gru" or "lstm".')
    cell_unit = tf.nn.rnn_cell.DropoutWrapper(cell_unit, output_keep_prob=keep_prob)
    if FLAGS.num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_unit] * FLAGS.num_layers)
    else:
        cell = cell_unit
    return cell

def predict(input_data, cell, state, keep_prob):
    """ Build the unrolled RNN prediction graph. """
    dim = FLAGS.hidden_size
    # embedding layer
    embedding = _variable_on_cpu('embedding', [FLAGS.vocab_size, dim])
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    inputs = tf.nn.dropout(inputs, keep_prob)

    # rnn layer
    input_list = tf.unpack(tf.transpose(inputs, perm=[1,0,2]))
    outputs, new_state = tf.nn.rnn(cell, input_list, initial_state=state)

    # softmax layer
    outputs = tf.reshape(tf.concat(1, outputs), shape=[-1, dim])
    softmax_w = tf.get_variable('softmax_w', shape=[dim, FLAGS.vocab_size])
    softmax_b = tf.get_variable('softmax_b', shape=[FLAGS.vocab_size])
    logits = tf.nn.bias_add(tf.matmul(outputs, softmax_w), softmax_b)
    return (logits, new_state)

def loss(logits, targets):
    """ Build the graph to evaluate loss. """
    cost = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])],
        weights=[tf.ones([FLAGS.batch_size * FLAGS.num_steps])])
    cost_per_seq = tf.reduce_sum(cost) / FLAGS.batch_size / FLAGS.num_steps
    return cost_per_seq

def train_batch(loss, lr):
    """ Build the graph to train on a batch of data. """
    opt = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.max_grad_norm)
    train_op = opt.apply_gradients(zip(grads, tvars))

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    return train_op

def _variable_on_cpu(name, shape):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape)
    return var