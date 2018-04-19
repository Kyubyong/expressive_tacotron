# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

import sys
import os
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, load_vocab
from modules import *
from networks import transcript_encoder, reference_encoder, decoder1, decoder2
from utils import *

class Graph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        is_training=True if mode=="train" else False

        # Graph
        # Data Feeding
        # x: Text. int32. (N, Tx) or (32, 188)
        # y: Reduced melspectrogram. float32. (N, Ty//r, n_mels*r) or (32, ?, 400)
        # z: Magnitude. (N, Ty, n_fft//2+1) or (32, ?, 1025)
        # ref: Melspectrogram of Reference audio. float32. (N, Ty, n_mels) or (32, ?, 80)
        if mode=="train":
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
            self.ref = tf.reshape(self.y, (hp.batch_size, -1, hp.n_mels))
        else: # Synthesize
            self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.Tx))
            self.y = tf.placeholder(tf.float32, shape=(hp.batch_size, None, hp.n_mels*hp.r))
            self.ref = tf.placeholder(tf.float32, shape=(hp.batch_size, None, hp.n_mels))

        # Get encoder/decoder inputs
        self.transcript_inputs = embed(self.x, len(hp.vocab), hp.embed_size) # (N, Tx, E)
        self.reference_inputs = tf.expand_dims(self.ref, -1)

        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)

        # Networks
        with tf.variable_scope("net"):
            # Encoder
            self.texts = transcript_encoder(self.transcript_inputs, is_training=is_training) # (N, Tx=188, E)
            self.prosody = reference_encoder(self.reference_inputs, is_training=is_training) # (N, 128)
            self.prosody = tf.expand_dims(self.prosody, 1) # (N, 1, 128)
            self.prosody = tf.tile(self.prosody, (1, hp.Tx, 1)) # (N, Tx=188, 128)
            self.memory = tf.concat((self.texts, self.prosody), -1) # (N, Tx, E+128)

            # Decoder1
            self.y_hat, self.alignments = decoder1(self.decoder_inputs,
                                                     self.memory,
                                                     is_training=is_training) # (N, T_y//r, n_mels*r)
            # Decoder2 or postprocessing
            self.z_hat = decoder2(self.y_hat, is_training=is_training) # (N, T_y//r, (1+n_fft//2)*r)

        # monitor
        self.audio = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)

        if mode=="train":
            # Loss
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            self.loss = self.loss1 + self.loss2

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss2'.format(mode), self.loss2)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            tf.summary.image("{}/mel_gt".format(mode), tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode), tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode), tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode), tf.expand_dims(self.z_hat, -1), max_outputs=1)

            tf.summary.audio("{}/sample".format(mode), tf.expand_dims(self.audio, 0), hp.sr)
            self.merged = tf.summary.merge_all()
         
if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")
    
    sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    with sv.managed_session() as sess:

        if len(sys.argv) == 2:
            sv.saver.restore(sess, sys.argv[1])
            print("Model restored.")

        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, gs = sess.run([g.train_op, g.global_step])

                # Write checkpoint files
                if gs % 1000 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))

                    # plot the first alignment for logging
                    al = sess.run(g.alignments)
                    plot_alignment(al[0], gs)

            if gs > hp.num_iterations:
                break

    print("Done")
