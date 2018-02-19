'''
    This script handles generating batches of messages - response (in Int IDs)
'''

import os
import pickle
import random

from batchStructure import Batch


# some macros
data_store = '/home/paperspace/Desktop/tc_nlp/dict_data'
file_name = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
dataset_path = os.path.join(data_store, file_name)
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


def loadData(dataset_path):
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples


def loadBatches(data, total_size, batch_size):
    for i in range(0, total_size, batch_size):
        yield data[i:min(total_size, i + batch_size)]


def processBatch(unprocessed_batch):
    """
    This function pad the input messages into equal length, and store them into the batchStructure.
    :return:
    """
    processed_batch = Batch()
    processed_batch.inputMsgLength = [len(pair[0]) for pair in unprocessed_batch]
    processed_batch.outputResponseLength = [len(pair[1]) for pair in unprocessed_batch]

    max_input_length = max(processed_batch.inputMsgLength)
    max_output_length = max(processed_batch.outputResponseLength)

    for pair in unprocessed_batch:
        msg = list(pair[0])
        processed_batch.inputMsgIDs.append(msg + [PAD_TOKEN] * (max_input_length - len(msg)))

        resp = list(pair[0])
        processed_batch.outputResponseIDs.append(msg + [PAD_TOKEN] * (max_output_length - len(resp)))

    return processed_batch


def generateBatches(data, batch_size):
    """
    Generate batches of data, all in integer IDs
    :return:
    """
    random.shuffle(data)
    batches = []
    size = len(data)
    for unprocessed_batch in loadBatches(data, size, batch_size):
        processed_batch = processBatch(unprocessed_batch)
        batches.append(processed_batch)
    return batches


