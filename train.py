from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import model
import text_input

tf.app.flags.DEFINE_string('data_dir', './data/tinyshakespeare/', 'Data directory')
tf.app.flags.DEFINE_string('data_file', 'input.txt', 'Data filename')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Dir to save checkpoint file and summary')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'Number of epochs to train')
tf.app.flags.DEFINE_float('init_scale', 0.01, 'Initial scale of all parameters')
tf.app.flags.DEFINE_float('dropout', 0, 'Dropout rate for LSTM outputs, 0 is no dropout')
tf.app.flags.DEFINE_float('init_lr', 5e-3, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'Learning rate decay rate')
tf.app.flags.DEFINE_integer('decay_after', 3, 'LR decays after this number of epochs')
tf.app.flags.DEFINE_integer('log_steps', 10, 'The step interval to print log to stdout')
tf.app.flags.DEFINE_integer('summary_steps', 10, 'The step interval to write summary')
tf.app.flags.DEFINE_integer('save_epochs', 1, 'The epoch interval to save model')

FLAGS = tf.app.flags.FLAGS

def train():
    print "Building training graph ..."
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("char-rnn", initializer=initializer):
            keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
            cell = model.build_cell(keep_prob)

            inputs = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_steps], name='inputs')
            targets = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_steps], name='targets')
            lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            initial_state = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, cell.state_size], name='initial_state')

            logits, final_state = model.predict(inputs, cell, initial_state, keep_prob)
            loss = model.loss(logits, targets)
            train_op = model.train_batch(loss, lr)

        # create saver and summary
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        # load data
        print "Loading data ..."
        reader = text_input.TextReader(os.path.join(FLAGS.data_dir, FLAGS.data_file))
        reader.prepare_data()
        train_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'train.cPickle'), FLAGS.batch_size, FLAGS.num_steps)
        test_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'test.cPickle'), FLAGS.batch_size, FLAGS.num_steps)

        total_steps = FLAGS.num_epochs * train_loader.num_batch
        save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        zero_state = cell.zero_state(FLAGS.batch_size, dtype=tf.float32).eval(session=sess)
        global_step = 0

        def eval(sess, loader, state):
            test_loss = 0.
            for _ in xrange(loader.num_batch):
                x_batch, y_batch = loader.next_batch()
                feed = {inputs: x_batch, targets: y_batch, keep_prob: 1., initial_state: state}
                state, loss_value = sess.run([final_state, loss], feed_dict=feed)
                test_loss += loss_value
            return test_loss / loader.num_batch

        # training
        for epoch in xrange(FLAGS.num_epochs):
            current_lr = FLAGS.init_lr * (FLAGS.lr_decay ** (max(epoch - FLAGS.decay_after + 1, 0)))
            state = zero_state
            training_loss = 0.
            for _ in xrange(train_loader.num_batch):
                global_step += 1
                start_time = time.time()
                x_batch, y_batch = train_loader.next_batch()
                feed = {inputs: x_batch, targets: y_batch, keep_prob: (1.-FLAGS.dropout), lr: current_lr, initial_state: state}
                state, loss_value, _ = sess.run([final_state, loss, train_op], feed_dict=feed)
                duration = time.time() - start_time
                training_loss += loss_value

                if global_step % FLAGS.log_steps == 0:
                    format_str = ('%s: step %d/%d (epoch %d/%d), loss = %.2f (%.3f sec/batch), lr: %.5f')
                    print(format_str % (datetime.now(), global_step, total_steps, epoch+1, FLAGS.num_epochs, loss_value,
                        duration, current_lr))

                if global_step % FLAGS.summary_steps == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step)

            if epoch % FLAGS.save_epochs == 0:
                saver.save(sess, save_path, global_step)
            train_loader.reset_pointer()

            # epoch summary
            training_loss /= train_loader.num_batch
            summary_writer.add_summary(_summary_for_scalar('training_loss', training_loss), global_step)
            test_loss = eval(sess, test_loader, zero_state)
            test_loader.reset_pointer()
            summary_writer.add_summary(_summary_for_scalar('test_loss', test_loss), global_step)
            print("Epoch %d: training_loss = %.2f, test_loss = %.2f" % (epoch+1, training_loss, test_loss))

def _summary_for_scalar(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()