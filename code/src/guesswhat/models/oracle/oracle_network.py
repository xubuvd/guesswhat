import tensorflow as tf

from neural_toolbox import rnn, utils
from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features

class OracleNetwork(ResnetModel):

    def __init__(self, config, num_words, device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            embeddings = []
            self.batch_size = None

            #Sources: is_training, question, seq_length, category, spatial, answer
            # QUESTION
            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = utils.get_embedding(self._question,
                                           n_words=num_words,
                                           n_dim=int(config['model']['question']["embedding_dim"]),
                                           scope="word_embedding")
            #word_emb shape = [batch_size, max_seq, 300]

            lstm_states, _ = rnn.variable_length_LSTM(\
                            word_emb,
                            num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                            seq_length=self._seq_length
                        )
            #lstm_states shape = [batch_size, 512]
            embeddings.append(lstm_states)

            # CATEGORY
            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')
                
                cat_emb = utils.get_embedding(\
                            self._category,
                            int(config['model']['category']["n_categories"]) + 1,  # we add the unkwon category
                            int(config['model']['category']["embedding_dim"]),
                            scope="cat_embedding")
                """
                cat_emb shape = [batch_size,512]
                """
                embeddings.append(cat_emb)

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)

            # IMAGE
            if config['inputs']['image']:
                self._image = tf.placeholder(tf.float32,
                            [self.batch_size] + config['model']['image']["dim"],
                            name='image')
                self.image_out = get_image_features(
                    image=self._image, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['model']['image']
                )
                embeddings.append(self.image_out)

            # CROP
            if config['inputs']['crop']:
                self._crop = tf.placeholder(tf.float32,\
                        [self.batch_size] + config['model']['crop']["dim"],\
                        name='crop')
                self.crop_out = get_image_features(
                    image=self._crop, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config["model"]['crop'])
                embeddings.append(self.crop_out)

            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)
            #emb shape = [batch_size, 512+512+8]
            
            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                l1 = utils.fully_connected(emb, num_hiddens, activation='relu', scope='l1')
                #l1 shape = [batch_size, 512]
                self.pred = utils.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                #pred = [batch_size,3]
                self.best_pred = tf.argmax(self.pred, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.pred, self._answer))
            self.error = tf.reduce_mean(utils.error(self.pred, self._answer))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

