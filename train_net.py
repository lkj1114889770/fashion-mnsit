
"""
 -*- coding: utf-8 -*-
 @author: Kaijian Liu
 @email:1114889770@qq.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from resnet import resnet_v2
from CNNnet import cnn_net
import random

tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data/fashion'
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


tf.app.flags.DEFINE_string('net', None, 'net to use for train')
FLAGS = tf.app.flags.FLAGS

def cnn_model_fn(features, labels, mode):
    if FLAGS.net == 'resnet_v2':
        logits = resnet_v2(features['x'],mode == tf.estimator.ModeKeys.TRAIN)
        print('use', FLAGS.net, '------------------')
    elif FLAGS.net == 'cnnnet':
        logits = cnn_net(features['x'],mode)
        print('use', FLAGS.net, '------------------')
    else:
        raise ValueError('Not a valid parmeter:', FLAGS.net)
    # logits = resnet_v2(features['x'],mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(0.001,tf.train.get_global_step(),100,0.95,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = input_data.read_data_sets(DATA_DIR)
    train_data = mnist.train.images  # Returns np.array
    images =[]
    #images = np.zeros((len(train_data)*3,28,28),dtype=np.float32)
    lables =[]
    for i in range(len(train_data)):
        image = np.reshape(train_data[i],(28,28))
        images.append(image)
        lables.append(mnist.train.labels[i])

        images.append(np.reshape(train_data[i][::-1],(28,28))[::-1])  # horizon flip
        lables.append(mnist.train.labels[i])

    train_data = np.array(images)
    print('Done!',np.shape(train_data),'-------------------------------------')
    train_labels = np.asarray(lables, dtype=np.int32)
    train_data, train_labels = shuffle(train_data, train_labels)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./'+FLAGS.net+'_model/')

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for j in range(50):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1000)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

if __name__ == "__main__":
    tf.app.run()