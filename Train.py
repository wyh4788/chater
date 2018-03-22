


'''
 THis is the script to train
'''

import tensorflow as tf
from batch_generation import *
from chatbot import Seq2seqModel
import math
import os

##################### Control Panel ###########################

hidden_size = 1024
num_hidden_layers = 2
word_embedding_size = 1024
#vocab_size
encoder_dropout_rate_placeholder = 0.0
Train = True
Decode = False
beam_size = 1
learning_rate = 0.0001
numEpochs = 30
batch_size = 128
steps_per_checkpoint = 2
load_from_prev = True
use_beam_search = False

##################### Control Panel ###########################

print ("I am using tensorflow version of: ")
print (str(tf.__version__))

log_dir = './my_log_dir'
data_store = '/home/paperspace/Desktop/tc_nlp/dict_data'
file_name = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
dataset_path = os.path.join(data_store, file_name)
model_dir_path = "/home/paperspace/Desktop/tc_nlp/trained_model"
model_name = "Trained_today"

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
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    current_step = 0
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    for e in range(numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, numEpochs))
        batches = generateBatches(trainingSamples, batch_size)
        for c, nextBatch in enumerate(batches):
            # loss, summary = model.train(sess, nextBatch)
            loss = model.train(sess, nextBatch)
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                #tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                print ("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                # summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(model_dir_path, model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)