# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def transcript_encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, Tx, E], with dtype of int32. Text inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of text hidden vectors. Has the shape of (N, Tx, E).
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, Tx, E/2)
        
        # Encoder CBHG 
        ## Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, Tx, K*E/2)
        
        ## Max pooling
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, Tx, K*E/2)
          
        ## Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, Tx, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, Tx, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # (N, Tx, E/2) # residual connections
          
        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, Tx, E/2)

        ## Bidirectional GRU
        texts = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, Tx, E)
    
    return texts


def reference_encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of (N, Ty, n_mels), with dtype of float32.
                Melspectrogram of reference audio.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Prosody vectors. Has the shape of (N, 128).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # 6-Layer Strided Conv2D -> (N, T/64, n_mels/64, 128)
        tensor = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn1")

        tensor = tf.layers.conv2d(inputs=tensor, filters=32, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn2")

        tensor = tf.layers.conv2d(inputs=tensor, filters=64, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn3")

        tensor = tf.layers.conv2d(inputs=tensor, filters=64, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn4")

        tensor = tf.layers.conv2d(inputs=tensor, filters=128, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn5")

        tensor = tf.layers.conv2d(inputs=tensor, filters=128, kernel_size=3, strides=2, padding='SAME')
        tensor = bn(tensor, is_training=is_training, activation_fn=tf.nn.relu, scope="bn6")

        # Unroll -> (N, T/64, 128*n_mels/64)
        N, _, W, C = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, (N, -1, W*C))

        # GRU -> (N, T/64, 128) -> (N, 128)
        tensor = gru(tensor, num_units=128, bidirection=False, scope="gru")
        tensor = tensor[:, -1, :]

        # FC -> (N, 128)
        prosody = tf.layers.dense(tensor, 128, activation=tf.nn.tanh)

    return prosody

def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, Ty/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, Tx, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, Ty/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, Ty/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, Ty/r, E)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, Ty/r, E)
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, Ty/r, E)
          
        # Outputs => (N, Ty/r, n_mels*r)
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)
    
    return mel_hats, alignments

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, Ty/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted linear spectrogram tensor with shape of [N, Ty, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, Ty, E*K/2)
         
        # Max pooling
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, Ty, E*K/2)

        ## Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, Tx, E/2)
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, Tx, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) # (N, Ty, E/2)
         
        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, Ty, E/2)
         
        # Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, bidirection=True) # (N, Ty, E)
        
        # Outputs => (N, Ty, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs
