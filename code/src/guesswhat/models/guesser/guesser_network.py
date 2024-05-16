import tensorflow as tf
from neural_toolbox import rnn, utils
from generic.tf_utils.abstract_network import AbstractNetwork

class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False,policy_gradient=False):
        AbstractNetwork.__init__(self, "guesser", device=device)
        alpha = 0.70
        self.c = 1.1
        mini_batch_size = None

        with tf.variable_scope(self.scope_name, reuse=reuse):
            self.num_words = num_words
            self.config = config

            # Training
            self.is_training = tf.placeholder_with_default(True, shape=(), name="is_training")

            # Dialogues
            self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
            self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')
            
            self.dialog_3d = tf.placeholder(tf.int32, [mini_batch_size, None,None], name='dialog_3d')#[batch_size, max_qnum, max_qa_len]
            self.dialog_3d_answer = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialog_3d_answer')#[batch_size, max_qnum]
            self.dialog_question_num = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialog_question_num')#[batch_size,max_qnum]
            self.max_qnum = tf.placeholder(tf.int32, shape=[], name='max_qnum')

            # Objects
            self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
            self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cats')
            self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="targets_index")
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size], name='cum_rewards')

            # Parameters
            zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
            zero_state = tf.tile(zero_state, [tf.shape(self.dialog_3d)[0], 1])  # trick to do a dynamic size 0 tensors
            self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_c")
            self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_h")

            self.object_cats_emb = utils.get_embedding(
                self.obj_cats,
                config['no_categories'] + 1,
                config['cat_emb_dim'],
                scope='cat_embedding')
            self.objects_input = tf.concat([self.object_cats_emb, self.obj_spats], axis=2)
            # objects_input [mini_batch_size,max_objs,8+256]
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])
            # flat_objects_inp [mini_batch_size*max_objs, 8 + 256]
            with tf.variable_scope('obj_mlp'):
                h1 = utils.fully_connected(
                    self.flat_objects_inp,
                    n_out=config['obj_mlp_units'],
                    activation='swish',
                    scope='l1')
                #h1 shape = [mini_batch_size*max_objs,512]
                #h1_drop = tf.cond(self.is_training,lambda:tf.nn.dropout(h1,keep_prob=self.config['keep_prob']),lambda:h1)
                """
                h2 = utils.fully_connected(
                    h1,
                    n_out=config['dialog_emb_dim'],
                    activation='tanh',
                    scope='l2')
                """
                #h2 shape = [mini_batch_size*max_objs,512]
            self.obj_embs = tf.reshape(h1, [-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])#(batch_size, max_objs, 512) 
            self.dialog_3d_trans = tf.transpose(self.dialog_3d,perm=[1,0,2])#[max_qnum,batch_size,max_qa_len]
            self.dialog_3d_answers_trans = tf.transpose(self.dialog_3d_answer,perm=[1,0])#[max_qnum, batch_size]
            self.softmax,facts_sequence,progressive_loss = self.VDST()#(batch_size, max_objs),(batch_size*self.max_qnum, max_objs)
            self.predicted_sequence = tf.identity(facts_sequence)
            self.selected_object = tf.argmax(self.softmax, axis=1)#row
            #selected_object shape = [mini_batch_size,]

            #self.targets (mini_batch_size,)  rank(targets) = 1
            #self.targets = tf.Print(self.targets,[tf.rank(self.targets)],"beforetargets: ")
            targets_sequence = tf.tile(tf.expand_dims(self.targets,axis=1),[1,self.max_qnum])
            targets_sequence = tf.reshape(targets_sequence,[-1])
            #targets_sequence = tf.Print(targets_sequence,[tf.rank(targets_sequence)],"targets_sequence: ")

            self.cross_entropy_loss = utils.cross_entropy(facts_sequence, targets_sequence)
            #self.loss = tf.reduce_mean(progressive_loss)
            self.loss = alpha*tf.reduce_mean(self.cross_entropy_loss) + (1.0 - alpha)*tf.reduce_mean(progressive_loss)
            self.error = tf.reduce_mean(utils.error(self.softmax, self.targets))
            if policy_gradient:
                with tf.variable_scope('policy_gradient_loss'):
                    self.log_of_policy = tf.identity(self.cross_entropy_loss)
                    self.policy_gradient_loss = alpha*tf.reduce_mean(self.log_of_policy) + (1.0 - alpha)*tf.reduce_mean(progressive_loss)
                    self.loss = self.policy_gradient_loss
    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

    def VDST(self):
        batch_size = tf.shape(self.dialog_3d_trans)[1]#self.config['batch_size']
        max_qlen = tf.shape(self.dialog_3d_trans)[-1]
        def cond(j,prev_p,c,h,vlist,p_subtraction):return tf.less(j,self.max_qnum)
        def body(j,prev_p,c,h,vlist,p_subtraction):
            q_j = self.dialog_3d_trans[j,:,:]#[max_qnum,batch_size,max_qlen] -> [batch_size,max_qlen]
            a_j = self.dialog_3d_answers_trans[j,:]#[max_qnum,batch_size] -> [batch_size,]
            
            qj_emb = utils.get_embedding(q_j,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)
            aj_emb = utils.get_embedding(a_j,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)
            
            attention = tf.reduce_sum(tf.multiply(self.obj_embs,tf.expand_dims(prev_p,-1)), axis=1, keepdims=False)#(batch_size, 512)
            qj_vj = tf.concat([qj_emb,tf.tile(tf.expand_dims(attention,axis=1),[1,max_qlen,1])],axis=2)#(batch_size, max_qlen,512+512)
            
            step_length = self.dialog_question_num[:,j]
            state_c,state_h = self.Encoder(c=c,h=h,x=qj_vj,seq_len=step_length,reuse=True)
            qa_j = tf.concat([state_h,aj_emb],axis=1)#(batch_size, 512+512) [q_j; a_j]
            curr_p = self.CMM_Simplified(qa_j,prev_p,reuse=True)#(batch_size, max_objs)
            vlist = tf.concat([vlist,curr_p],axis=1)
            p_subtraction += utils.cross_entropy(self.c + curr_p - prev_p, self.targets)
            return tf.add(j, 1),curr_p,state_c,state_h,vlist,p_subtraction

        #self.obj_mask (batch_size, max_objs)
        init_prob_0 = self.obj_mask*tf.ones_like(self.obj_mask, dtype=tf.float32)
        init_prob = tf.div(init_prob_0,tf.reduce_sum(init_prob_0,axis=1,keepdims=True))#(batch_size, max_objs)
        #init_prob = tf.Print(init_prob,[init_prob,tf.rank(init_prob)],"Memory-init_prob: ")
        q1 = self.dialog_3d_trans[0,:,:]#[batch_size, max_qlen]
        a1 = self.dialog_3d_answers_trans[0,:]#[batch_size,]
        q1_emb = utils.get_embedding(q1,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=False)#(batch_size, max_qlen, 512)
        a1_emb = utils.get_embedding(a1,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)#(batch_size, 512)
        step_length = self.dialog_question_num[:,0]#(batch_size,)
        attention = tf.reduce_sum(tf.multiply(self.obj_embs,tf.expand_dims(init_prob,-1)), axis=1, keepdims=False)#(batch_size, 512)
        q1_v1 = tf.concat([q1_emb,tf.tile(tf.expand_dims(attention,axis=1),[1,max_qlen,1])],axis=2)#(batch_size, max_qlen,512+512)
        c,h = self.Encoder(c=self.decoder_zero_state_c,
                    h=self.decoder_zero_state_h,
                    x=q1_v1,
                    seq_len=step_length,reuse=False)
        #c,h (batch_size, 512)
        qa_1 = tf.concat([h,a1_emb],axis=1)#(batch_size, 512+512)   [q_j; a_j]
        curr_p = self.CMM_Simplified(qa_1,init_prob,reuse=False)#(batch_size, max_objs)
        p_subtraction = utils.cross_entropy(self.c + curr_p - init_prob, self.targets)#(batch_size,)
        vlist = curr_p#(batch_size, max_objs)
        j = tf.constant(1)
        _,facts,_,_,vlist,loss_subtraction = tf.while_loop(cond, body,
                    [j,curr_p,c,h,vlist,p_subtraction],
                    shape_invariants=[\
                        j.get_shape(),\
                        curr_p.get_shape(),\
                        c.get_shape(),\
                        h.get_shape(),tf.TensorShape([None,None]),p_subtraction.get_shape()])
        #self.facts (batch_size, max_objs)
        vlist = tf.reshape(vlist,[-1,tf.shape(self.obj_mask)[1]]) #(batch_size*self.max_qnum, max_objs)
        #vlist (batch_size, max_objs) of size self.max_qnum, (batch_size, max_objs*self.max_qnum)
        return facts,vlist,loss_subtraction

    def attention(self,prob,scope_name="selfAtten",reuse=False):
        """
        args:
            prob: probablity on objects in an image, (batch_size, max_objs)
            obj,  object representation, (batch_size, max_objs, 512)
        returns:
            self.visual, (batch_size, 512)
        """
        pass

    def Encoder(self,c,h,x,seq_len,reuse=False,scope_name="GuesserEncoder"):
        """
        returns:
            c, (batch_size, 512)
            h, (batch_size, 512)
            output, (batch_size, max_qlen, 512)
        """
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
            scope=scope_name)
        return state.c,state.h

    def CMM_Simplified(self,h,prev_p,scope_name="CMM",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
        """
        pi = softmax(tanh(Uo * Vh)/sqrt(d))
        args:
            self.obj_embs, (batch_size, max_objs, 512)
            h, representation of question-answer pair, (batch_size, 512+512)
            prev_p, (batch_size, max_objs)
        return:
            p_norm, (batch_size, max_objs)
        """
        with tf.variable_scope(scope_name, reuse=reuse):
            batch_size = tf.shape(self.obj_embs)[0]#.get_shape()[0].value
            max_objs = tf.shape(self.obj_embs)[1]
            object_feature_size = self.obj_embs.get_shape()[2].value
            scale_emb_size = 128

            #update of object representation
            curr_obj_embs = tf.multiply(self.obj_embs,tf.expand_dims(prev_p,-1))#(batch_size, max_objs, 512)
            #batch_size = tf.Print(batch_size,[batch_size,max_objs,object_feature_size],"CMM-1: ")
            #CMM-1: [64][20][512]

            #CMM version 3
            Vh = utils.fully_connected(h,512,use_bias=True,activation='tanh',reuse=reuse,scope="CMM_Vh")#(batch_siz,512)
            odot_Obj_h = tf.multiply(tf.expand_dims(Vh,axis=1),curr_obj_embs)#(batch_size,max_objs,512)
            concat_odot_h_obj = tf.concat([odot_Obj_h,tf.tile(tf.expand_dims(Vh,axis=1),[1,max_objs,1]),curr_obj_embs],axis=2)#(batch_size,max_objs,512+512+512)
            concat_odot_h_obj_reshape = tf.reshape(concat_odot_h_obj,[-1,object_feature_size*3])#(batch_size*max_objs, 512)
            #lay1 = utils.fully_connected(concat_odot_h_obj_reshape,512,use_bias=True,activation='tanh',reuse=reuse,scope="CMM_layer_1")
            lay2 = utils.fully_connected(concat_odot_h_obj_reshape,128,use_bias=True,activation='tanh',reuse=reuse,scope="CMM_layer_2")
            lay3 = utils.fully_connected(lay2,1,use_bias=True,reuse=reuse,activation=None,scope="CMM_layer_3")#(batch_size * max_objs,1)
            logits = tf.reshape(lay3,[batch_size,max_objs])

            def masked_softmax(scores, mask):
                # subtract max for stability
                scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
                # compute padded softmax
                exp_scores = tf.exp(scores)
                exp_scores = exp_scores*mask + 1e-10
                exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
                return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])
            p = masked_softmax(logits, self.obj_mask)#(batch_size, max_objs)

            accum_p = tf.multiply(p,prev_p) + 1e-5#(batch_size, max_objs)
            p_norm = tf.div(accum_p,tf.reduce_sum(accum_p, axis=1,keepdims=True))#(batch_size, max_objs)
            return p_norm

