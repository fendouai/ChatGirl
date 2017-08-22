#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

train_x=np.load('./idx_q.npy', mmap_mode='r')
train_y=np.load('./idx_a.npy', mmap_mode='r')
print(train_x.shape)

batch_size=128
sequence_length=20
num_encoder_symbols=10000
num_decoder_symbols=10000
embedding_size=100
learning_rate=0.001

encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])

logits=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length,num_decoder_symbols])
targets=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
weights=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length])


train_weights=np.ones(shape=[batch_size,sequence_length],dtype=np.float32)

cell=tf.nn.rnn_cell.BasicLSTMCell(sequence_length)

def seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size):
	encoder_inputs = tf.unstack(encoder_inputs, axis=0)
	decoder_inputs = tf.unstack(decoder_inputs, axis=0)
	results,states=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    output_projection=None,
    feed_previous=False,
    dtype=None,
    scope=None
)
	return results

def get_loss(logits,targets,weights):
	loss=tf.contrib.seq2seq.sequence_loss(
		logits,
		targets=targets,
		weights=weights
	)
	return loss

results=seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size)
logits=tf.stack(results,axis=0)
print(logits)
loss=get_loss(logits,targets,weights)

pred = tf.argmax(logits, 2)
correct_pred = tf.equal(tf.cast(pred, tf.int64), tf.cast(targets, tf.int64))
print("correct_pred.shape",correct_pred.shape)
accuracy =tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("accuracy.shape",accuracy.shape)

train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

saver=tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	count=0
	while(count<100):
		count=count+1
		print("cout:",count)
		for step in range(0,1000):
			print("step:",step)
			train_encoder_inputs=train_x[step*batch_size:step*batch_size+batch_size,:]
			train_targets=train_y[step*batch_size:step*batch_size+batch_size,:]
			train_decoder_inputs = np.zeros([batch_size, sequence_length])
			#results_value=sess.run(results,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_decoder_inputs})
			cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
			                                 weights:train_weights,decoder_inputs:train_decoder_inputs})
			print(cost)
			accuracy_value=sess.run(accuracy, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
			                                 weights:train_weights,decoder_inputs:train_decoder_inputs})
			print(accuracy_value)
			op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
			                                 weights: train_weights, decoder_inputs: train_decoder_inputs})
			step=step+1

	saver.save(sess, "./model/model.ckpt")