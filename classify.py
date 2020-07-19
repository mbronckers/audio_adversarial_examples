## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

import argparse

import scipy.io.wavfile as wav

import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # set TF logging level (1/2/3)

import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"



def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()
    
    with tf.Session() as sess:
        
        if args.input.split(".")[-1] == 'mp3':
            raw = pydub.AudioSegment.from_mp3(args.input)
            audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
        
        elif args.input.split(".")[-1] == 'wav':
            sample_rate, audio = wav.read(args.input)
            
        else:
            raise Exception("Unknown file format")
        
        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])
        
        max_chars_per_second = 50
        samples_per_character = sample_rate / max_chars_per_second
        length = (N-1) // samples_per_character

        print("-"*80)
        print("Sample rate:", sample_rate)
        print("Audio: ", audio)
        print("Length of audio: ", N)
        print("Length in characters: ", length)
        print("-"*80)

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)


        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path) # restore DeepSpeech model

        # logits shape [max_time, batch_size, num_classes]
        print('Logits shape: ', logits.shape)
        
        # note to self: greedy decoder is beam with top_widths=1 and beam_width=1
        # decoded.values is a CTCBeamSearchDecoder
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)
        
        # returns sparse tensor of classified characters
        # .indices: idx w/ non-zero values (2D)
        # .values: values for each of the idx in indices (1D)
        # .dense_shape: dense shape of the sparse tensor (non-explicit zeros)
        output = sess.run(decoded, {new_input: [audio], # run CTCBeamSearch decoder on audio with length len(audio)//320
                                   lengths: [length]})

        print("-"*80)
        print("Classification:")
        print("".join([toks[x] for x in output[0].values])) 
        print("-"*80)

main()
