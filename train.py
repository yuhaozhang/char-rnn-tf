from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import model
import input

tf.app.flags.DEFINE_string('data_dir', './data/tinyshakespeare/', 'Data directory')
tf.app.flags.DEFINE_string('data_file', 'input.txt', 'Data directory')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Dir to save checkpoint file and summary')
tf.app.flags.DEFINE_integer('max_train_steps', 5500, 'Max number of steps to train')
tf.app.flags.DEFINE_integer('init_scale', 0.01, 'Initial scale of all parameters')

FLAGS = tf.app.flags.FLAGS

LOG_STEP_INTERVAL = 10
SUMMARY_STEP_INTERVAL = 10
SAVE_STEP_INTERVAL = 1000

INITIAL_LR = 0.01
LR_DECAY_RATE = 0.9
DECAY_STEPS = 100

TRAIN_DROPOUT_RATE = 1.0

def train():
    print "Building training graph ..."
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("char-rnn", initializer=initializer):
            global_step = tf.Variable(0, trainable=False)

            inputs = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_steps], name='inputs')
            targets = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_steps], name='targets')
            keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
            lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            initial_state = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.hidden_size*FLAGS.num_layers*2], name='initial_state')

            logits, final_state = model.predict(inputs, initial_state, keep_prob)
            loss = model.loss(logits, targets)
            train_op = model.train_batch(loss, global_step, lr)

        # create saver and summary
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

        # load data
        print "Loading training data ..."
        reader = input.TextReader(os.path.join(FLAGS.data_dir, FLAGS.data_file))
        reader.prepare_data()
        loader = input.DataLoader(os.path.join(FLAGS.data_dir, 'train.cPickle'), FLAGS.batch_size, FLAGS.num_steps)

        for step in xrange(FLAGS.max_train_steps):
            current_lr = INITIAL_LR * (LR_DECAY_RATE ** (step // DECAY_STEPS))
            start_time = time.time()
            x_batch, y_batch = loader.next_batch()
            if step == 0:
                state = np.zeros(initial_state.get_shape())
            dict_to_feed = {inputs: x_batch, targets: y_batch, keep_prob: TRAIN_DROPOUT_RATE, lr: current_lr, initial_state: state}
            state, loss_value, _ = sess.run([final_state, loss, train_op], feed_dict=dict_to_feed)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model loss is NaN.'

            if step % LOG_STEP_INTERVAL == 0:
                seqs_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d/%d, loss = %.2f (%.1f seqs/sec; %.3f sec/batch), lr: %.5f')
                print (format_str % (datetime.now(), step, FLAGS.max_train_steps, loss_value, seqs_per_sec, 
                    sec_per_batch, current_lr))

            if step % SUMMARY_STEP_INTERVAL == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % SAVE_STEP_INTERVAL == 0:
                save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, save_path, global_step=step)


def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()