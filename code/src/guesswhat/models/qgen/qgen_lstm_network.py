import tensorflow as tf
import numpy as np
import logging
from neural_toolbox import utils
from generic.tf_factory.attention_factory import get_attention
from generic.tf_utils.abstract_network import AbstractNetwork
from generic.data_provider.nlp_utils import padder

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class QGenNetworkLSTM(AbstractNetwork):
    #TODO: add dropout
    def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)
        self.logger = logging.getLogger()
        self.config = config
        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            mini_batch_size = None

            # Image
            self.images = tf.placeholder(tf.float32, [mini_batch_size] + config['image']["dim"], name='images')
            '''
            baseline: self.images shape=[batch_size,VGG16_fc8=1000]
            mytest:   self.images shape=[batch_size,VGG16_conv5_3=14,14,512]
            self.images shape=[batch_size,VGG16_pool5=7,7,512]
            [batch_size,36,2048+8]
            '''

            self.num_words = num_words
            # Question
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')#[batch_size, max_seq_len]
            #self.dialogues = 
            #<start> w1_1,w1_2,w1_3 a1 w2_1,w2_2,w2_3,w2_4,w2_5 a2 w3_1,w3_2,w3_3,w3_4,w3_5 <stop> 0 0 0 0 of length max_seq_len

            self.answer_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='answer_mask')  # 1 if keep and (1 q/a 1) for (START q/a STOP)
            
            self.dialog_3d = tf.placeholder(tf.int32, [mini_batch_size, None,None], name='dialog_3d')#[batch_size, max_qnum, max_qa_len]
            #self.dialog_3d = questions matrix, not contains the <start> or answer position
            #w1_1,w1_2,w1_3 0    0    0
            #w2_1,w2_2,w2_3,w2_4,w2_5 0
            #w3_1,w3_2,w3_3,w3_4,w3_5 0
            #0     0    0    0   0    0
            #0     0    0    0   0    0
            self.dialog_3d_answer = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialog_3d_answer')#[batch_size, max_qnum]
            self.dialog_question_num = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialog_question_num')#[batch_size,max_qnum]
            #self.dialog_question_num = [4 6 6 0 0]
            #<start> w1_1,w1_2,w1_3 = 4
            #a1, w2_1,w2_2,w2_3,w2_4,w2_5 = 6
            #a2, w3_1,w3_2,w3_3,w3_4,w3_5 = 6
            #0 padded question of length 0
            #0 padded question of length 0
            self.prev_question = tf.placeholder(tf.int32,[mini_batch_size,None],name="prev_question")
            self.prev_answer = tf.placeholder(tf.int32,[mini_batch_size,],name="prev_answer")
            self.prev_answer_inverse = tf.placeholder(tf.int32,[mini_batch_size,],name="prev_answer_inverse")

            self.prev_qlen = tf.placeholder(tf.int32, [mini_batch_size], name='prev_qlen')
            #self.objects_j = tf.placeholder(tf.float32, [mini_batch_size,config['object_num'],config['object_feature_size']], name='objects_j')
            #[batch_size, objects_num, object_feature_size]
            #self.v_j = tf.placeholder(tf.float32, [mini_batch_size,None], name='v_j')#[batch_size, object_feature_size]
            self.prob = tf.placeholder(tf.float32, [mini_batch_size,None], name='prob')#[batch_size,objects_nums]
            """
            self.answer_mask shape=[batch_size,max_length]
            [[1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1]
             [1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1]
             ....
             [1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1]]
             only zero on the pos of answer
            """
            self.padding_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='padding_mask')
            """
            self.padding_mask:shape= [batch_size,max_length]
            [[1,1,1,1,1,1,0,0,0,0]
             [1,1,1,1,1,1,1,0,0,0]
              ....
             [1,1,1,1,1,1,1,1,1,1]]
            only zero over the padding pos
            """
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')
            """
            self.seq_length = [29,30,.....30], a list of size batch_size
            """
            self.max_qnum = tf.placeholder(tf.int32, shape=[], name='max_qnum')

            #Rewards
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size, None], name='cum_reward')

            #DECODER Hidden state (for beam search)
            zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
            zero_state = tf.tile(zero_state, [tf.shape(self.images)[0], 1])  # trick to do a dynamic size 0 tensors

            self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_c")
            self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_h")
            decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.decoder_zero_state_c, h=self.decoder_zero_state_h)

            self.encoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="encoder_state_c")
            self.encoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="encoder_state_h")
            self.encoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.encoder_zero_state_c, h=self.encoder_zero_state_h)

            #Misc
            self.is_training = tf.placeholder_with_default(True, shape=(), name="is_training")#tf.placeholder(tf.bool, name='is_training')
            self.greedy = tf.placeholder_with_default(False, shape=(), name="greedy") # use for graph
            self.is_first_question = tf.placeholder(tf.bool, name='is_first_question')
            #self.consin_scalar_deepRL = tf.placeholder(tf.float32,shape=(), name='consin_scalar_deepRL')

            self.samples = None

            """
            self.dialogues = 
            [[start, w1,w2,w3,? a1 w4,w5,w6,w7? a2 ...Stop,pad,pad,pad,pad,pad,pad,pad]
             [start, w1,w2,w3,w4,?,a1,w5,w6,w7,w8,?,a2,w9,w10,...,stop,pad,pad,pad,pad]
             [start,w1,w2,w3,w4,w5,?,a1,w6,w7,w8,w9,?,a2,w10,w11,w12,w13,?,a3,... stop]
             ...
             [start ................................. stop,pad,pad,pad,pad,pad,pad,pad]]
             contains the single games of size batch_size.
            """
            # remove last token of <stop_dialogue>
            input_dialogues = self.dialogues[:, :-1]#(batch_size, max_seq_len-1)
            input_seq_length = self.seq_length - 1 #elment-wise minus

            # remove first token(=start token)
            rewards = self.cum_rewards[:, 1:]
            target_words = self.dialogues[:, 1:] #(batch_size, max_seq_len-1)

            # to understand the padding:
            # input (removed <stop_dialogue>)
            #   <start>  is   it   a    blue   <?>   <yes>   is   it  a    car  <?>   <no>
            # target
            #    is      it   a   blue   <?>    -      is    it   a   car  <?>   -   <stop_dialogue>  -
            
            #Reduce the embedding size of the image
            with tf.variable_scope('image_embedding'):
                images_reshape = tf.reshape(self.images,shape=[-1,config['image']["dim"][-1]])#(batch_size, 36, 2056)
                image_out = utils.fully_connected(images_reshape,config['object_feature_size'],reuse=False,activation="swish")#(batch_size,object_num,512)
                self.image_out = tf.reshape(image_out,[config['batch_size'],config['object_num'],config['object_feature_size']])
                #self.image_emb shape = [batch_size, 512]
            self.dialog_3d_trans = tf.transpose(self.dialog_3d,perm=[1,0,2])#[max_qnum,batch_size,max_qa_len]
            self.dialog_3d_answers_trans = tf.transpose(self.dialog_3d_answer,perm=[1,0])#[max_qnum, batch_size]

            self.decoder_output = self.VDST()#(batch_size, max_seq - 1, 1024)
            max_sequence = tf.reduce_max(self.seq_length)
            #compute the softmax for evaluation
            with tf.variable_scope('decoder_output'):
                flat_decoder_output = tf.reshape(self.decoder_output, [-1, self.config['num_lstm_units']])
                #flat_decoder_output shape = [batch_size*(max_seq - 1), 1024]
                flat_mlp_output = utils.fully_connected(flat_decoder_output, num_words)#[batch_size*(max_seq - 1), num_words]

                #retrieve the batch/dialogue format
                mlp_output = tf.reshape(flat_mlp_output, [tf.shape(self.seq_length)[0], max_sequence - 1, num_words])  # Ignore th STOP token

                #target_words shape = [batch_size, max_seq - 1]
                self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mlp_output, labels=target_words)
                #self.cross_entropy_loss shape = [batch_size, max_seq - 1]
            with tf.variable_scope('ml_loss'):
                ml_loss = tf.identity(self.cross_entropy_loss)
                ml_loss *= self.answer_mask[:, 1:]  # remove answers (ignore the <stop> token)
                ml_loss *= self.padding_mask[:, 1:]  # remove padding (ignore the <start> token)

                # Count number of unmask elements
                count = tf.reduce_sum(self.padding_mask) - tf.reduce_sum(1 - self.answer_mask[:, :-1]) - 1  # num_unpad - num_qa - START token

                ml_loss = tf.reduce_sum(ml_loss, axis=[1,0])  # reduce over dialogue dimension, column
                self.ml_loss = ml_loss / count  # Normalize
                self.loss = self.ml_loss
            if policy_gradient:
                with tf.variable_scope('rl_baseline'):
                    #self.decoder_output shape = [batch_size, max_seq - 1, 1024]
                    decoder_out = tf.stop_gradient(self.decoder_output)  # take the LSTM output (and stop the gradient!)

                    flat_decoder_output = tf.reshape(decoder_out, [-1, self.config['num_lstm_units']])#[batch_size*(max_seq - 1), 1024]

                    flat_h1 = utils.fully_connected(flat_decoder_output, n_out=128, activation='relu', scope='baseline_hidden')
                    flat_baseline = utils.fully_connected(flat_h1, 1, activation='relu', scope='baseline_out')

                    self.baseline = tf.reshape(flat_baseline, [tf.shape(self.seq_length)[0], max_sequence-1])
                    #self.baseline shape = [batch_size, max_seq - 1]

                    self.baseline *= self.answer_mask[:, 1:]
                    self.baseline *= self.padding_mask[:, 1:]

                with tf.variable_scope('policy_gradient_loss'):
                    #self.log_of_policy shape = [batch_size, max_seq - 1]
                    self.log_of_policy = tf.identity(self.cross_entropy_loss)
                    self.log_of_policy *= self.answer_mask[:, 1:]  # remove answers (<=> predicted answer has maximum reward) (ignore the START token in the mask)
                    # No need to use padding mask as the discounted_reward is already zero once the episode terminated

                    rewards *= self.answer_mask[:, 1:]
                    self.score_function = tf.multiply(self.log_of_policy, rewards - self.baseline)  # score function

                    self.baseline_loss = tf.reduce_sum(tf.square(rewards - self.baseline))
                    self.policy_gradient_loss = tf.reduce_mean(tf.reduce_sum(self.score_function, axis=1), axis=0)
                    self.loss = self.policy_gradient_loss

    def get_loss(self):
        return self.loss
    def get_accuracy(self):
        return self.loss
    def Decoder(self,c,h,x,seq_len,scope_name="word_decoder",reuse=False):
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            self.config['num_lstm_units'],
            layer_norm=False,
            dropout_keep_prob=1.0,
            reuse=reuse)
        initial_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        output,state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=x,
            initial_state=initial_state,
            sequence_length=seq_len,
            time_major=False,
            scope=scope_name
        )
        return state.c,state.h,output

    def VDST(self,scope="multi-step-reasoning"):
        """
        args:
            dialog_3d, [max_qnum, batch_size, max_qa_len], dialog questions
            dialog_question_num, [batch_size,max_qnum]
            dialog_3d_answer, [max_qnum, batch_size], dialog answers
            objects, [batch_size,objects_num,object_feature_size]
        returns:
            vlist, [max_qnum, batch_size, object_feature_size]
        """
        batch_size = self.config['batch_size']
        objects_num = self.config['object_num']
        object_feature_size = self.config["object_feature_size"]
        max_qlen = tf.shape(self.dialog_3d_trans)[-1]
        def cond(j,prev_p,vlist,c,h):return tf.less(j,self.max_qnum)
        def body(j,prev_p,vlist,c,h):
           
            self.UoOR(pi=prev_p,reuse=True)
            v_j = self.OsDA(h=self.config['multihead'],g=self.config['glimpse'],reuse=True)#(batch_size, 128)
            q_j = self.dialog_3d_trans[j,:,:]#[max_qnum,batch_size,max_qlen] -> [batch_size,max_qlen]
            a_j = self.dialog_3d_answers_trans[j,:]#[max_qnum,batch_size] -> [batch_size,]
            q_j_emb = utils.get_embedding(q_j,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=True)
            v_j_tile = tf.tile(tf.expand_dims(v_j,axis=1),[1,max_qlen,1])
            a_j_emb = utils.get_embedding(a_j,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=True)
            input_step_length = self.dialog_question_num[:,j]#[batch_size,]
            textual_visual_concat = tf.concat([q_j_emb,v_j_tile],axis=2)
            state_c,state_h,output = self.Decoder(
                c=c,h=h,
                x=textual_visual_concat,
                seq_len=input_step_length,reuse=True)
            h_qa = tf.concat([state_h,a_j_emb],axis=1)
            prob_j = self.CMM_Simplified(h_qa,prev_p,reuse=True)

            vlist = tf.concat([vlist,tf.expand_dims(output, axis=0)],axis=0)
            return tf.add(j, 1),prob_j, vlist, state_c,state_h

        #For the first question
        init_prob_0 = np.full((batch_size,objects_num), 1.0/float(objects_num), dtype=np.float32)
        self.UoOR(pi=init_prob_0,reuse=False)
        v1 = self.OsDA(h=self.config['multihead'],g=self.config['glimpse'],reuse=False)#(batch_size, object_feature_size)
        v1_tile = tf.tile(tf.expand_dims(v1,axis=1),multiples=[1,max_qlen,1])

        q1 = self.dialog_3d_trans[0,:,:]#[batch_size, max_qlen]
        a1 = self.dialog_3d_answers_trans[0,:]#[batch_size,]
        q1_emb = utils.get_embedding(q1,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=False)
        a1_emb = utils.get_embedding(a1,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=True)
        input_step_length = self.dialog_question_num[:,0]
        textual_visual_concat = tf.concat([q1_emb,v1_tile],axis=2)#(batch_size,max_qlen,object_feature_size+512)
        c,h,output = self.Decoder(
            c=self.decoder_zero_state_c,
            h=self.decoder_zero_state_h,
            x=textual_visual_concat,
            seq_len=input_step_length,
            reuse=False)
        #c,h = (batch_size, 1024)
        #output = (batch_size, max_qlen, 1024)
        h_qa = tf.concat([h, a1_emb],axis=1)#(bs,1024+512)

        prob_j = self.CMM_Simplified(h_qa,init_prob_0,reuse=False)
        
        vlist = tf.expand_dims(output, axis=0)#(1,batch_size, max_qlen, 1024)
        #For the remaining questiones
        j = tf.constant(1)
        _,_,vlist,_,_ = tf.while_loop(cond, body,\
                [j,prob_j,vlist,c,h],\
                shape_invariants=[\
                    j.get_shape(),\
                    prob_j.get_shape(),\
                    tf.TensorShape([None,None,None,None]),\
                    c.get_shape(),\
                    h.get_shape()])
        #vlist = vlist[1:,:,:,:]#[max_qnum,batch_size,max_qlen, 1024]

        # concat questiones to obtain the whole dialogue sequence
        max_seq_len = tf.shape(self.dialogues)[-1] - 1
        def slice_cond(i,result):return tf.less(i, batch_size)
        def slice_body(i,result):
            def subcond(j,output,jsum):return tf.less(j, self.max_qnum)
            def subbody(j,output,jsum):
                word_num = self.dialog_question_num[i,j]
                jsum += tf.cast(word_num,tf.int32)
                output = tf.concat([output,vlist[j,i,:,:][0:word_num,:]],axis=0)
                return tf.add(j, 1),output,jsum
            output = tf.zeros(shape=[1,self.config['num_lstm_units']],dtype=tf.float32)
            j = tf.constant(0)
            jsum = tf.constant(0)
            _,output,jsum = tf.while_loop(subcond,subbody,[j,output,jsum],shape_invariants=[j.get_shape(),tf.TensorShape([None,None]),jsum.get_shape()])
            output = tf.reshape(output[1:,:],shape=[-1,self.config['num_lstm_units']])
            padd_size = max_seq_len - tf.cast(jsum,tf.int32)
            zero_state = tf.zeros([padd_size, self.config['num_lstm_units']],dtype=tf.float32)
            output = tf.concat([output,zero_state],axis=0)
            result = tf.concat([result,tf.expand_dims(output,axis=0)],axis=0)
            return tf.add(i, 1),result
        i = tf.constant(0)
        result = tf.zeros(shape=(1, max_seq_len, self.config['num_lstm_units']), dtype=tf.float32)
        _,result = tf.while_loop(slice_cond,slice_body,[i,result],shape_invariants=[i.get_shape(),tf.TensorShape([None,None,None])])
        result = result[1:,:,:]#(batch_size,max_seq-1,1024)
        return result

    def build_sampling_graph(self, config, tokenizer, max_length=12):
        
        if self.samples is not None: return
        
        def prepaire_first_question():
           
            self.UoOR(pi=self.prob,reuse=True)
            init_v = self.OsDA(h=config['multihead'],g=config['glimpse'],reuse=True)#(batch_size, object_feature_size)
            return init_v,self.prob
        
        def prepaire_next_question():
            
            prev_a_emb = utils.get_embedding(self.prev_answer,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=True)
            prev_h_Q = tf.concat([self.decoder_zero_state_h,prev_a_emb],axis=1)
            self.UoOR(pi=self.prob,reuse=True)#TODO here have a bug, ought to reuse the prev representation of objects set that generated the self.prev_question.
            prob_j = self.CMM_Simplified(prev_h_Q,self.prob, reuse=True)
            
            self.UoOR(pi=prob_j,reuse=True)
            #the current salient visual feature for next question
            v_j = self.OsDA(h=config['multihead'],g=config['glimpse'],reuse=True)#(batch_size, object_feature_size)
            return v_j,prob_j

        # define stopping conditions
        def stop_cond(states_c, states_h, tokens, seq_length, stop_indicator,v_j_plus_1,prob_j):

            has_unfinished_dialogue = tf.less(tf.shape(tf.where(stop_indicator))[0],tf.shape(stop_indicator)[0]) # TODO use "any" instead of checking shape
            has_not_reach_size_limit = tf.less(tf.reduce_max(seq_length), max_length)

            return tf.logical_and(has_unfinished_dialogue,has_not_reach_size_limit)

        # define one_step sampling
        with tf.variable_scope(self.scope_name):
            stop_token = tf.constant(tokenizer.stop_token)#?
            stop_dialogue_token = tf.constant(tokenizer.stop_dialogue)#<stop>

        def step(prev_state_c, prev_state_h, tokens, seq_length, stop_indicator,v_j_plus_1,prob_j):
            """
            tokens    = [max_seq, batch_size]
            prev_q_a  = [batch_size, max_qa_len]
            image_out = [batch_size, object_num,feture_size]
            v = [batch_size, feature_size]
            h = [batch_size,hidden_size] = c
            """
            input = tf.gather(tokens, tf.shape(tokens)[0] - 1)#[1,batch_size]
            #input: the last token in this batch

            # Look for new finish dialogue
            is_stop_token = tf.equal(input, stop_token)#[1,batch_size]
            is_stop_dialogue_token = tf.equal(input, stop_dialogue_token)
            is_stop = tf.logical_or(is_stop_token, is_stop_dialogue_token)
            stop_indicator = tf.logical_or(stop_indicator, is_stop)  #[1,batch_size]

            # increment seq_length when the dialogue is not over
            seq_length = tf.where(stop_indicator, seq_length, tf.add(seq_length, 1))#[1,batch_size]

            # compute the next words. TODO: factorize with qgen.. but how?!
            with tf.variable_scope(self.scope_name, reuse=True):
                word_emb = utils.get_embedding(
                    input,
                    n_words=tokenizer.no_words,
                    n_dim=config['word_embedding_size'],
                    scope="word_embedding",
                    reuse=True)#[batch_size, 512]
                inp_emb = tf.concat([word_emb,v_j_plus_1],axis=1)#[batch_size,512+512]
                #inp_emb = tf.concat([word_emb, self.image_emb], axis=1)#[batch_size,1024]
                with tf.variable_scope("word_decoder"):
                    lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                        config['num_lstm_units'],
                        layer_norm=False,
                        dropout_keep_prob=1.0,
                        reuse=True)
                    state = tf.contrib.rnn.LSTMStateTuple(c=prev_state_c, h=prev_state_h)
                    out, state = lstm_cell(inp_emb, state)
                    #out = [batch_size, hidden_size]
                    #state = [c, h], c=[batch_size, hidden_size], h=[batch_size, hidden_size]

                    # store/update the state when the dialogue is not finished (after sampling the <?> token)
                    cond = tf.greater_equal(seq_length, tf.subtract(tf.reduce_max(seq_length), 1))
                    state_c = tf.where(cond, state.c, prev_state_c)
                    state_h = tf.where(cond, state.h, prev_state_h)

                with tf.variable_scope('decoder_output'):
                    output = utils.fully_connected(state_h, tokenizer.no_words, reuse=True)#[batch_size, num_words]
                    sampled_tokens = tf.cond(self.greedy,\
                                        lambda: tf.argmax(output, axis=1),\
                                        lambda: tf.reshape(tf.multinomial(logits=output,num_samples=1), [-1]))#[batch_size,]
                    #to control the sampling,to encourage the model to explore in the action space
                    sampled_tokens = tf.cast(sampled_tokens, tf.int32)#[batch_size,]
            """
            tokens shape = [max_seq, batch_size]
            """
            tokens = tf.concat([tokens, tf.expand_dims(sampled_tokens, 0)], axis=0)
            return state_c, state_h, tokens, seq_length, stop_indicator,v_j_plus_1,prob_j

        # initialialize sequences
        batch_size = tf.shape(self.seq_length)[0]
        seq_length = tf.fill([batch_size], 0)
        stop_indicator = tf.fill([batch_size], False)

        def LastDecisonMaking():
            prev_a_emb = utils.get_embedding(self.prev_answer,n_words=self.num_words,n_dim=self.config['word_embedding_size'],scope="word_embedding",reuse=True)
            prev_h_Q = tf.concat([self.decoder_zero_state_h,prev_a_emb],axis=1)
            self.UoOR(pi=self.prob,reuse=True)
            last_prob = self.CMM_Simplified(prev_h_Q,self.prob,reuse=True)
            return last_prob

        def FirstDecisonMaking():return self.prob

        with tf.variable_scope(self.scope_name, reuse=True):
            v_j_plus_1,prob_j = tf.cond(self.is_first_question,
                lambda: prepaire_first_question(),
                lambda: prepaire_next_question())
            
            self.guessing_prob = tf.cond(self.is_first_question,
                lambda:FirstDecisonMaking(),
                lambda:LastDecisonMaking())
        transpose_dialogue = tf.transpose(self.dialogues, perm=[1,0])#(max_seq, batch_size)

        self.samples = tf.while_loop(stop_cond, step, [self.decoder_zero_state_c,\
                                                    self.decoder_zero_state_h,\
                                                    transpose_dialogue,\
                                                    seq_length,\
                                                    stop_indicator,\
                                                    v_j_plus_1,\
                                                    prob_j],
                                     shape_invariants=[self.decoder_zero_state_c.get_shape(),\
                                                    self.decoder_zero_state_h.get_shape(),\
                                                    tf.TensorShape([None, None]),\
                                                    seq_length.get_shape(),\
                                                    stop_indicator.get_shape(),\
                                                    v_j_plus_1.get_shape(),\
                                                    prob_j.get_shape()])

    def UoOR(self,pi,scope_name="bn",reuse=False):
        with tf.variable_scope(scope_name):
            self.norm_x = tf.multiply(self.image_out,tf.expand_dims(pi,-1))# + self.image_out
            #self.norm_x = tf.contrib.layers.batch_norm(\
            #    tf.multiply(self.image_out,tf.expand_dims(pi,-1)) + self.image_out,\
            #    updates_collections=None,decay=0.99,epsilon=1e-5,scale=True,fused=True,data_format="NHWC",\
            #    is_training=self.is_training,reuse=reuse,scope="contribn")
    
    def OsDA_AblationStudy(self,h=1,g=2,scope_name="VisReasonAblationStudy",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
        with tf.variable_scope(scope_name, reuse=reuse):
            batch_size = self.config['batch_size']#64
            objects_num = self.config['object_num']#36
            object_feature_size = self.config['object_feature_size']#300
            v_output = tf.reduce_mean(self.norm_x, axis=1, keep_dims=False)
            return v_output

    def OsDA(self,h=1,g=2,scope_name="Self_Difference_Attention",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
        """
        VisualReasoning
        Visual information driven Reasoning
        """
        with tf.variable_scope(scope_name, reuse=reuse):
            batch_size = self.config['batch_size']#64
            objects_num = self.config['object_num']#36
            object_feature_size = self.config['object_feature_size']#300
            norm_x_flatten = tf.reshape(self.norm_x, shape=[batch_size,objects_num*object_feature_size])
            #(batch_size, objects_num, objects_num*object_feature_size)
            obj_diff = tf.multiply(tf.tile(self.norm_x,[1,1,objects_num]),\
                tf.subtract(tf.tile(self.norm_x,[1,1,objects_num]),tf.expand_dims(norm_x_flatten,axis=1)))#(3,4,20)
            out_v = list()
            for head in range(h):
                W = tf.get_variable("WH_{}".format(head),[objects_num*object_feature_size,g],initializer=weight_initializer)
                b = tf.get_variable("bH_{}".format(head), [g,],initializer=tf.zeros_initializer())
               
                #(64,36,36*512)*(36*512,g)=(64,36,g) -> (64,g,36)
                logits = tf.transpose(tf.einsum('ijk,kl->ijl',obj_diff,W) + b,[0,2,1])#(3,4,20)*(20,2) = (3,4,2)->(3,2,4) (bs,object_num,g)
                p = tf.nn.softmax(tf.rsqrt(1.0*object_feature_size)*logits)#(bs,g,objects_num)
                #VisualReasoning-logits: [[3.17183423 -5.57640123 2.23528337...]...][64 2 36]
                #VisualReasoning-p: [[0.285181493 4.52700515e-05 0.111784734...]...][64 2 36]

                v = tf.reshape(tf.matmul(p, self.norm_x),[batch_size,-1])#(3,2,4)*(3,4,5) = (3,2,5) -> (3,2*5)
                out_v.append(v)
            v_cat = tf.concat(out_v,axis=1)#(bs, g*object_feature_size*h)
            v_output = utils.fully_connected(v_cat,object_feature_size,reuse=reuse,scope="mlp_1")
            return v_output #(batch_size, object_feature_size)

    def CMM_Simplified(self,h,prev_p,scope_name="Linguistic_Reasoning",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
        """
        pi = softmax(tanh(Uo * Vh)/sqrt(d))
        """
        with tf.variable_scope(scope_name, reuse=reuse):
            batch_size = self.norm_x.get_shape()[0].value
            object_feature_size = self.norm_x.get_shape()[-1].value
            scale_emb_size = 300
            
            U = tf.get_variable("MLB_U",[object_feature_size,scale_emb_size],initializer=weight_initializer)
            Uo = tf.tanh(tf.einsum('ijk,kl->ijl',self.norm_x,U))#(batch_size,objects_num,scale_emb_size)
            
            Vh = utils.fully_connected(h,scale_emb_size,use_bias=False,activation='tanh',reuse=reuse,scope="MLB_Vh")#(batch_size,scale_emb_size)
            
            Uo_odot_Vh = tf.multiply(tf.expand_dims(Vh,axis=1),Uo)#(batch_size,objects_num, object_feature_size)
            Uo_odot_Vh_reshape = tf.reshape(Uo_odot_Vh,[-1,object_feature_size])
            
            Uo_odot_Vh_MLP = utils.fully_connected(Uo_odot_Vh_reshape,1,use_bias=True,reuse=reuse,scope="MLB_Nonlinear")#(-1,)
            Uo_odot_Vh_MLP = tf.reshape(Uo_odot_Vh_MLP,[batch_size,-1,1])
            
            logits = tf.squeeze(Uo_odot_Vh_MLP)#(batch_size,36)
            p = tf.nn.softmax(tf.rsqrt(1.0*scale_emb_size)*logits)#(batch_size, objects_num)
            
            accum_p = tf.multiply(p,prev_p) + 1e-5#(batch_size, objects_num)
            p_norm = tf.div(accum_p,tf.reduce_sum(accum_p, axis=1,keepdims=True))#(batch_size, objects_num)
            return p_norm
    def CMM(self,h,prev_p,scope_name="Linguistic_Reasoning",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
        """
        LinguisticReasoning for VQA problem
        Linguistic information driven Reasoning
        pi = softmax(Vh*tanh(Uo * Vh)/sqrt(d))
        args:
            h, linguistic information, (batch_size, 1024)
            prev_p, previous prob. distribution over objects of number 36, (batch_size,36)
        returns:
            curr_p, the current prob. distribution over objects of number 36 after the current question with answer
        """
        with tf.variable_scope(scope_name, reuse=reuse):
            batch_size = self.norm_x.get_shape()[0].value
            object_feature_size = self.norm_x.get_shape()[-1].value#self.config['object_feature_size']
            scale_emb_size = 300
 
            U = tf.get_variable("MLB_U",[object_feature_size,scale_emb_size],initializer=weight_initializer)
            Uo = tf.tanh(tf.einsum('ijk,kl->ijl',self.norm_x,U))#(batch_size,objects_num,scale_emb_size)
            Vh = utils.fully_connected(h,scale_emb_size,use_bias=False,activation='tanh',reuse=reuse,scope="MLB_Vh")#(batch_size,scale_emb_size)
            
            odot = tf.multiply(tf.expand_dims(Vh,axis=1),Uo)#(batch_size,objects_num, object_feature_size)
            odot_re = tf.reshape(odot,[-1,object_feature_size])
            odot_MLB_P_bias = utils.fully_connected(odot_re,scale_emb_size,use_bias=True,reuse=reuse,scope="MLB_Nonlinear")#(-1,)
            odot_MLB = tf.reshape(odot_MLB_P_bias,[batch_size,-1,scale_emb_size])
            logits = tf.squeeze(tf.matmul(odot_MLB, tf.expand_dims(Vh,axis=-1)))#(bs,36)
            p = tf.nn.softmax(tf.rsqrt(1.0*scale_emb_size)*logits)#(batch_size, objects_num)
            
            accum_p = tf.multiply(p,prev_p) + 1e-5#(batch_size, objects_num)
            p_norm = tf.div(accum_p,tf.reduce_sum(accum_p, axis=1,keepdims=True))#(batch_size, objects_num)
            return p_norm

