import tensorflow as tf

from neural_toolbox import rnn, utils

from generic.tf_utils.abstract_network import AbstractNetwork


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        mini_batch_size = None

        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Dialogues
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

            # Objects
            self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
            self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cats')
            self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="targets_index")

            self.object_cats_emb = utils.get_embedding(
                self.obj_cats,
                config['no_categories'] + 1,
                config['cat_emb_dim'],
                scope='cat_embedding')

            self.objects_input = tf.concat([self.object_cats_emb, self.obj_spats], axis=2)
            #self.objects_input shape = [mini_batch_size,max_objs,8+256]
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])
            #self.flat_objects_inp shape = [mini_batch_size*max_objs, 8 + 256]

            with tf.variable_scope('obj_mlp'):
                h1 = utils.fully_connected(
                    self.flat_objects_inp,
                    n_out=config['obj_mlp_units'],
                    activation='relu',
                    scope='l1')
                #h1 shape = [mini_batch_size*max_objs,512]
                h2 = utils.fully_connected(
                    h1,
                    n_out=config['dialog_emb_dim'],
                    activation='relu',
                    scope='l2')
                #h2 shape = [mini_batch_size*max_objs,512]
            obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])
            #obj_embs shape = [mini_batch_size, max_objs, 512]
            
            # Compute the word embedding
            input_words = utils.get_embedding(self.dialogues,
                                              n_words=num_words,
                                              n_dim=config['word_emb_dim'],
                                              scope="input_word_embedding")
            #input_words shape = [mini_batch_size, max_seq, 512]
            
            last_states, _ = rnn.variable_length_LSTM(input_words,
                                               num_hidden=config['num_lstm_units'],
                                               seq_length=self.seq_length)
            #last_states shape = [mini_batch_size, 512]

            last_states = tf.reshape(last_states, [-1, config['num_lstm_units'], 1])
            #last_states shape = [mini_batch_size, 512, 1]
            
            scores = tf.matmul(obj_embs, last_states)
            #scores shape = [mini_batch_size, max_objs, 1]

            scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])
            #scores shape = [mini_batch_size, max_objs]

            def masked_softmax(scores, mask):
                # subtract max for stability
                scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
                # compute padded softmax
                exp_scores = tf.exp(scores)
                exp_scores *= mask
                exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
                return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])

            #self.obj_mask shape = [mini_batch_size, max_objs]
            self.softmax = masked_softmax(scores, self.obj_mask)
            #self.softmax shape = [mini_batch_size, max_objs]

            self.selected_object = tf.argmax(self.softmax, axis=1)
            #self.selected_object shape = [mini_batch_size,]
            
            self.loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self.targets))
            self.error = tf.reduce_mean(utils.error(self.softmax, self.targets))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

