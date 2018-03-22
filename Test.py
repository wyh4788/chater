
'''
 THis is the script to Test
'''

import tensorflow as tf
from batch_generation import *
from chatbot import Seq2seqModel
import math
import os
import numpy as np
import sys

##################### Control Panel ###########################

hidden_size = 1024
num_hidden_layers = 2
word_embedding_size = 1024
#vocab_size
encoder_dropout_rate_placeholder = 0.0
Train = False
Decode = True
beam_size = 3
learning_rate = 0.0001
numEpochs = 30
batch_size = 128
steps_per_checkpoint = 2
load_from_prev = True
use_beam_search = True

##################### Control Panel ###########################

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
        This is a method copied from GitHub Repo: seq2seq_chatbot_new
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))

model_dir_path = "/home/paperspace/Desktop/tc_nlp/trained_model"  # copied from Train.py


word2id, id2word, trainingSamples = loadData(dataset_path)
total_size = len(trainingSamples)
print('The training size is of %d ' % total_size)
vocab_size = len(word2id)
print('The vocab size is of %d ' % vocab_size)
with tf.Session() as sess:
    model = Seq2seqModel(hidden_size, num_hidden_layers, vocab_size, word_embedding_size, encoder_dropout_rate_placeholder,
                 Train, Decode, use_beam_search, beam_size, word2id, learning_rate)
    ckpt = tf.train.get_checkpoint_state(model_dir_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and load_from_prev:
        print('Reloading model parameters..')
        print('==============================')
        print(ckpt.model_checkpoint_path)
        print('==============================')

        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Error: No Saved Model to Retrieve!')
    sys.stdout.write(">")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        predicted_ids = model.inference(sess, batch)
        # print(predicted_ids)
        print("The response is: ")
        predict_ids_to_seq(predicted_ids, id2word, beam_size)
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()