"""
    This script builds a chatbot using seq2seq model


    ####################################################################################################################
    1. sequence_mask dtype: the output of mask should be float or boolean? Which one more efficient
    2. Output_layer vs self-defined final fc multiplication
    3.
    ####################################################################################################################

"""




import tensorflow as tf
from tensorflow.python.util import nest

class Seq2seqModel:
    def __init__(self, hidden_size, num_hidden_layers, vocab_size, word_embedding_size, encoder_dropout_rate_placeholder,
                 Train, Decode, use_beam_search, beam_size, word_to_idx, learning_rate):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.word_embedding_size = word_embedding_size
        self.encoder_dropout_rate_placeholder = encoder_dropout_rate_placeholder
        self.Train = Train
        self.Decode = Decode
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.word_to_idx = word_to_idx
        self.learning_rate = learning_rate
        self.max_gradient_norm = 5.0
        print('===========================================================')
        print('Model Parameters: ')
        print("Hidden_size: " + str(self.hidden_size))
        print('num_hidden_layers: ' + str(self.num_hidden_layers))
        print('vocab_size: ' + str(self.vocab_size))
        print('encoder_dropout_rate_placeholder: ' + str(self.encoder_dropout_rate_placeholder))
        print('beam_size: ' + str(self.beam_size))
        print('learning_rate: ' + str(self.learning_rate))
        print('===========================================================')

        # building the graph
        self.encoder_output, self.encoder_states = self.build_encoder()
        self.build_decoder()


    def _create_rnn_cell(self):
        def single_rnn_cell():
            tf_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            #add dropout
            cell_dropout = tf.contrib.rnn.DropoutWrapper(tf_cell, output_keep_prob=1.0-self.encoder_dropout_rate_placeholder)
            return cell_dropout
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_hidden_layers)])
        return cell

    def build_encoder(self):
        print ("~~~~~~~~~~~~~~~~~~~~~~ Building the encoder ~~~~~~~~~~~~~~~~~~~~~~~~")
        # placeholders for the encoder end
        self.input_msg_IDs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.input_msg_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.encoder_dropout_rate_placeholder = tf.placeholder_with_default(input=0.0, shape=(), name='encoder_dropout_rate')

        # Text embedding
        with tf.variable_scope('WORD_EMBEDDING'):
            self.emb_matrix = tf.get_variable(name='embedding_matrix',
                                         shape=[self.vocab_size, self.word_embedding_size])
                                         # initializer=tf.contrib.layers.xavier_initializer(),
                                         # trainable=True)

        with tf.variable_scope('ENCODER'):
            encoder_cell = self._create_rnn_cell()
            encoder_input_wordvec = tf.nn.embedding_lookup(params=self.emb_matrix,
                                                           ids=self.input_msg_IDs,
                                                           name='input_message_embedding')
            encoder_output, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                               inputs=encoder_input_wordvec,
                                                               sequence_length=self.input_msg_length,
                                                               dtype=tf.float32)
            ##############################################################################################
            '''
            There is no dropout applied on the encoded vector (encoder_output). Maybe add it?
            '''
            ##############################################################################################
        return encoder_output, encoder_states


    def build_decoder(self):
        self.outputResponseIDs = tf.placeholder(tf.int32, [None, None], name='decoder_outputs')
        self.outputResponseLength = tf.placeholder(tf.int32, [None], name='decoder_outputs_length')
        self.max_output_length = tf.reduce_max(self.outputResponseLength, name='max_decoder_length')
        #print (self.max_output_length)

        self.mask = tf.sequence_mask(self.outputResponseLength, self.max_output_length, dtype=tf.float32, name='mask')
        with tf.variable_scope('DECODER'):

            # We would like beam search
            # First, copy the several variables 'beam_size' times
            if self.use_beam_search:
                encoder_outputs_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_output, multiplier=self.beam_size)
                encoder_input_length = tf.contrib.seq2seq.tile_batch(self.input_msg_length, multiplier=self.beam_size)
                encoder_state_tiled = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), self.encoder_states)
            self.batch_size = self.batch_size * self.beam_size

            # define "attention"
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size,
                                                                       memory=self.encoder_output,
                                                                       memory_sequence_length=self.input_msg_length)

            #################
            # Need to redefine attention mechanism to fit the need of beam search
            #################
            decoder_cell = self._create_rnn_cell()
            decoder_cell_attention = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                                         attention_mechanism=attention_mechanism,
                                                                         attention_layer_size=self.hidden_size,
                                                                         name='Attention_Wrapper')

            # initial state for the decoder
            decoder_initial_state = decoder_cell_attention.zero_state(batch_size=self.batch_size,
                                                                      dtype=tf.float32).clone(cell_state=self.encoder_states)
            fc_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))

            if self.Train == True:
                ending = tf.strided_slice(self.outputResponseIDs, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.emb_matrix, decoder_input)

                # print ("==========")
                # print (decoder_inputs_embedded)
                # print("==========")
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.outputResponseLength,
                                                                    time_major=False, name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell_attention, helper=training_helper,
                                                                   initial_state=decoder_initial_state, output_layer=fc_layer)

                # print (self.max_output_length)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                    maximum_iterations=self.max_output_length)

                self.build_trainer(decoder_outputs)

            elif self.Decode == True:
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<go>']
                end_token = self.word_to_idx['<eos>']
                if self.use_beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell_attention,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             embedding=self.emb_matrix,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=fc_layer)
                ########
                # else #
                ########
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                          maximum_iterations=10)
                if self.use_beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    def build_trainer(self, decoder_outputs):
        self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
        # print (self.decoder_predict_train)
        # print (self.decoder_logits_train)
        # print (self.outputResponseIDs)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                     targets=self.outputResponseIDs, weights=self.mask)
        # tf.summary.scalar('loss', self.loss)
        # self.summary_op = tf.summary.merge_all()

        # Optimizer Handle
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))



        # Training without max grad op
        # self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def train(self, sess, batch):

        feed_dict = {self.input_msg_IDs: batch.inputMsgIDs,
                      self.input_msg_length: batch.inputMsgLength,
                      self.outputResponseIDs: batch.outputResponseIDs,
                      self.outputResponseLength: batch.outputResponseLength,
                      self.encoder_dropout_rate_placeholder: 0.0,
                      self.batch_size: len(batch.inputMsgIDs)}
        # _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def inference(self, sess, batch):
        feed_dict = {self.input_msg_IDs: batch.inputMsgIDs,
                     self.input_msg_length: batch.inputMsgLength,
                     self.encoder_dropout_rate_placeholder: 0.0,
                     self.batch_size: len(batch.inputMsgIDs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict
