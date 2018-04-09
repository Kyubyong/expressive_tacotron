# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav, load_spectrograms
from scipy.io.wavfile import write
import os
import sys
from glob import glob
import numpy as np
from math import ceil


def looper(ref, start, batch_size):
    num = int(ceil(float(ref.shape[0]) / batch_size)) + 1
    tiled = np.tile(ref, (num, 1, 1))[start:start + batch_size]
    return tiled, start + batch_size % ref.shape[0]


def synthesize():
    if not os.path.exists(hp.sampledir):
        os.mkdir(hp.sampledir)

    # Load data
    texts = load_data(mode="synthesize")

    # pad texts to multiple of batch_size
    texts_len = texts.shape[0]
    num_batches = int(ceil(float(texts_len) / hp.batch_size))
    padding_len = num_batches * hp.batch_size - texts_len
    texts = np.pad(
        texts, ((0, padding_len), (0, 0)), 'constant', constant_values=0
    )

    # reference audio
    mels, maxlen = [], 0
    files = glob(hp.ref_audio)
    for f in files:
        _, mel, _ = load_spectrograms(f)
        mel = np.reshape(mel, (-1, hp.n_mels))
        maxlen = max(maxlen, mel.shape[0])
        mels.append(mel)

    ref = np.zeros((len(mels), maxlen, hp.n_mels), np.float32)
    for i, m in enumerate(mels):
        ref[i, :m.shape[0], :] = m

    # Load graph
    g = Graph(mode="synthesize")
    print("Graph loaded")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if len(sys.argv) == 1:
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored latest checkpoint")
        else:
            saver.restore(sess, sys.argv[1])
            print("Restored checkpoint: %s" % sys.argv[1])

        batches = [
            texts[i:i + hp.batch_size]
            for i in range(0, texts.shape[0], hp.batch_size)
        ]
        start = 0
        batch_index = 0
        # Feed Forward
        for batch in batches:
            ref_batch, start = looper(ref, start, hp.batch_size)
            ## mel
            y_hat = np.zeros(
                (batch.shape[0], 200, hp.n_mels * hp.r), np.float32
            )  # hp.n_mels*hp.r
            for j in tqdm.tqdm(range(200)):
                _y_hat = sess.run(
                    g.y_hat, {g.x: batch,
                              g.y: y_hat,
                              g.ref: ref_batch}
                )
                y_hat[:, j, :] = _y_hat[:, j, :]
            ## mag
            mags = sess.run(g.z_hat, {g.y_hat: y_hat})
            for i, mag in enumerate(mags):
                index_label = batch_index * hp.batch_size + i + 1
                if index_label > texts_len:
                    break
                print("File {}.wav is being generated ...".format(index_label))
                audio = spectrogram2wav(mag)
                write(
                    os.path.join(hp.sampledir, '{}.wav'.format(index_label)),
                    hp.sr, audio
                )

            batch_index += 1


if __name__ == '__main__':
    synthesize()
    print("Done")
