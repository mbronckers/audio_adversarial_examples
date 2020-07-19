## tf_logits.py -- end-to-end differentable text-to-speech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import time
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # set TF logging level (1/2/3)
sys.path.append("DeepSpeech")

import DeepSpeech

def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    # 0. Set digital signal processing parameters
    batch_size, signal_length = audio.get_shape().as_list() # get batch and signal_length by getting audio shape
    audio = tf.cast(audio, tf.float32) # cast audio file to float32 type
    num_mfcc_features = 26 # default n_input in DeepSpeech

    sample_rate = 16000 # 16KHz
    frame_size = 0.032 # 20ms to 40ms with 50% (+/-10%) overlap between consecutive frames
    frame_stride = 0.01 # 10ms stride 
    max_chars_per_second = 50
    samples_per_character = int(sample_rate / max_chars_per_second)

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    frame_length = int(round(frame_length)) # how many samples in a frame
    frame_step = int(round(frame_step)) 
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    print(frame_length)
    print(num_frames)

    # 1. Pre-emphasizer, a high-pass filter: passes only signals above a cutoff frequency and attenuates lower frequencies
    # source: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    pre_emphasis = 0.97
    audio = tf.concat((audio[:, :1], audio[:, 1:] - pre_emphasis*audio[:, :-1], \
                        np.zeros((batch_size, frame_length), dtype=np.float32)), 1)

    # 2. windowing into frames of frame_length samples, overlapping
    windowed = tf.stack([audio[:, i:i+frame_length] for i in range(0, signal_length - samples_per_character, samples_per_character)], 1)
    window = np.hamming(frame_length) # Hamming window
    windowed = windowed * window

    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [frame_length])
    ffted = 1.0 / frame_length * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2) + np.finfo(float).eps
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+np.finfo(float).eps

    # 5. Take the DCT to decorrelate the FFT coefficients
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:num_mfcc_features]

    # 6. Amplify high frequencies by sinusoidal liftering to the MFCCs to de-emphasize higher MFCCs 
    # (which has been claimed to improve speech recognition in noisy signals)
    _, nframes, ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff) 
    cep_lifter = 22 # liftering is filtering in the cepstral domain
    lift = 1 + (cep_lifter / 2.)*np.sin(np.pi*n/cep_lifter)
    feat *= lift
    width = feat.get_shape().as_list()[1]

    # 7. And now stick log energy next to the features
    feat = tf.concat( (tf.reshape(tf.log(energy), (-1, width, 1)), feat[:, :, 1:]), axis=2)

    return feat


def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.

    One frame is (window_width*num_mfcc_features) long
    """

    batch_size = new_input.get_shape()[0]
    context_width = 9 # number of frames in the context; default n_context in DeepSpeech
    num_mfcc_features = 26 # default n_input in DeepSpeech
    window_width = 2 * context_width + 1 # number of frames on both sides + ourselves

    # 1. Compute the MFCCs for the input audio
    # (this is differentable with our implementation above)
    empty_context = np.zeros((batch_size, context_width, num_mfcc_features), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. create overlapping windows and
    # remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, num_mfcc_feat]
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+window_width*num_mfcc_features]
                            for i in range(0, features.shape[1] - window_width*num_mfcc_features+1, num_mfcc_features)],1)
    features = tf.reshape(features, [batch_size, -1, window_width, num_mfcc_features])


    # 3. Finally we process it with DeepSpeech
    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)

        DeepSpeech.create_flags()
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    logits, _ = DeepSpeech.BiRNN(features, length, [0]*10)

    return logits
