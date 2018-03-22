from data_helpers import *
from batch_generation import  *
from testData import convertList2sentence
batch_size = 4

data_store = '/Users/wangshufan/Desktop/ADD/starterProject/chater/'
file_name = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
data_path = os.path.join(data_store, file_name)
word2id, id2word, trainingSamples = loadDataset(data_path)
trainingSamples = trainingSamples[0:20]


batches = getBatches(trainingSamples, batch_size)
writer = open("./OriginalBatchSample.txt", "w")
for c, nextBatch in enumerate(batches):
    assert len(nextBatch.encoder_inputs) == batch_size
    assert len(nextBatch.decoder_targets) == batch_size
    writer.write('Batch %d with %d examples \n' % (c, len(nextBatch.encoder_inputs)))
    for k in range(batch_size):
        input = 'Input: ' + convertList2sentence(nextBatch.encoder_inputs[k])
        output = 'Output:' + convertList2sentence(nextBatch.decoder_targets[k])
        writer.write(input + "    ||    " + output + '\n')
    writer.write('==================================\n')
writer.close()


data_store = '/Users/wangshufan/Desktop/ADD/starterProject/chater/'
file_name = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
data_path = os.path.join(data_store, file_name)
word2id, id2word, trainingSamples = loadDataset(data_path)
trainingSamples = trainingSamples[0:20]
batches = generateBatches(trainingSamples, batch_size)
writer2 = open("./MyBatchSample.txt", "w")
for c, nextBatch in enumerate(batches):
    assert len(nextBatch.inputMsgIDs) == batch_size
    assert len(nextBatch.outputResponseIDs) == batch_size
    writer2.write('Batch %d with %d examples \n' % (c, len(nextBatch.inputMsgIDs)))
    for k in range(batch_size):
        input = 'Input: ' + convertList2sentence(nextBatch.inputMsgIDs[k])
        output = 'Output:' + convertList2sentence(nextBatch.outputResponseIDs[k])
        writer2.write(input + "    ||    " + output + '\n')
    writer2.write('==================================\n')
writer2.close()