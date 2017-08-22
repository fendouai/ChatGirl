#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
from hello import Word_Id_Map


map=Word_Id_Map()
train_x=map.sentence2ids(["how","are","you","how","are","you","how","are","you","how","are","you"])
len_x=len(train_x)
pad_x=[0 for i in range(0,20-len_x)]
train_x=train_x+pad_x
train_x=np.asarray(train_x)
train_x=np.asarray([train_x])
print(train_x.shape)
print(train_x)
train_y=map.sentence2ids(['i','am','ok'])
len_y=len(train_y)
pad_y=[0 for i in range(0,20-len_y)]
train_y=train_y+pad_y
train_y=np.asarray(train_y)
train_y=np.asarray([train_y])
print(train_y.shape)
print(train_y)

batch_size=1
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
	saver.restore(sess, "./model/model.ckpt")
	train_encoder_inputs=train_x
	train_decoder_inputs=np.zeros([1,20])
	train_targets=train_y
	pred_value=sess.run(pred,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_targets})
	print(pred_value)
	sentence = map.ids2sentence(pred_value[0])
	print(sentence)
	accuracy_value=sess.run(accuracy, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
									 weights:train_weights,decoder_inputs:train_decoder_inputs})
	print(accuracy_value)