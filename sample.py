import os

import numpy as np
import tensorflow as tf

import model
import text_input

tf.app.flags.DEFINE_string('data_dir', './data/tinyshakespeare/', 'Data directory')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Dir to save checkpoint file and summary')
tf.app.flags.DEFINE_string('start_with', 'ROMEO', 'Characters to start with')
tf.app.flags.DEFINE_integer('max_length', 600, 'Number of characters to generate')

FLAGS = tf.app.flags.FLAGS

def sample():
    # build sampling graph
    with tf.variable_scope("char-rnn"):
        inputs = tf.placeholder(dtype=tf.int32, shape=[1,1], name='inputs')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        initial_state = tf.placeholder(dtype=tf.float32, shape=[1,FLAGS.hidden_size*FLAGS.num_layers*2], name='initial_state')
        logits, final_state = model.predict(inputs, initial_state, keep_prob)
    state = np.zeros(initial_state.get_shape())

    char2id = text_input.load_from_dump(os.path.join(FLAGS.data_dir, 'vocab.cPickle'))
    id2char = {v:k for k,v in char2id.items()}
    output_str = FLAGS.start_with

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            raise IOError('Cannot restore checkpoint file in ' + train_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for c in FLAGS.start_with[:-1]:
            x = np.array([[char2id[c]]])
            _, state = sess.run([logits, final_state], feed_dict={inputs:x, keep_prob:1., initial_state:state})

        last_id = char2id[FLAGS.start_with[-1]]
        for _ in xrange(FLAGS.max_length - len(FLAGS.start_with)):
            x = np.array([[last_id]])
            logits_value, state = sess.run([logits, final_state], feed_dict={inputs:x, keep_prob:1., initial_state:state})
            last_id = _sample_from_logits(logits_value)
            c = id2char[last_id]
            output_str += c
        print output_str

    return

def _sample_from_probs(logits):
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    cum = np.cumsum(probs)
    r = np.random.random()
    return np.argmax(cum>=r)

def main(_):
    if not tf.gfile.Exists(FLAGS.train_dir):
        raise Exception('Provided training directory does not exist.')
    sample()

if __name__ == '__main__':
    tf.app.run()
