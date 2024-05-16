import tensorflow as tf
from neural_toolbox import rnn, utils
from generic.tf_utils.abstract_network import AbstractNetwork

class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False,policy_gradient=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

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
            self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size], name='cum_reward')

            #Parameters
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
            #self.objects_input shape = [mini_batch_size,max_objs,8+256]
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])
            #self.flat_objects_inp shape = [mini_batch_size*max_objs, 8 + 256]
            with tf.variable_scope('obj_mlp'):
                h1 = utils.fully_connected(
                    self.flat_objects_inp,
                    n_out=config['obj_mlp_units'],
                    activation='tanh',
                    scope='l1')
                #h1 shape = [mini_batch_size*max_objs,512]
                h2 = utils.fully_connected(
                    h1,
                    n_out=config['dialog_emb_dim'],
                    activation='tanh',
                    scope='l2')
                #h2 shape = [mini_batch_size*max_objs,512]
            self.obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])
            #self.obj_embs = tf.reshape(h2, [config['batch_size'], -1, config['dialog_emb_dim']])
            #obj_embs shape = [batch_size, max_objs, 512]
           
            self.dialog_3d_trans = tf.transpose(self.dialog_3d,perm=[1,0,2])#[max_qnum,batch_size,max_qa_len]
            self.dialog_3d_answers_trans = tf.transpose(self.dialog_3d_answer,perm=[1,0])#[max_qnum, batch_size]
            self.Memory()
            #self.facts (max_qnum, batch_size, 512)
            self.Attention()#self.attended_fact (batch_size,512)
            
            #MemoryAttention version
            obj_embs_tr = tf.transpose(self.obj_embs,perm=[0,2,1])#(batch_size,512,max_objs)
            fact_attn = tf.expand_dims(self.attended_fact,axis=1)#(batch_size,1,512)
            #old scores = tf.matmul(obj_embs, last_states) #[mini_batch_size, max_objs, 1]
            scores = tf.matmul(fact_attn,obj_embs_tr)#(batch_size,1,max_objs)
            scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])
            #scores shape = [batch_size, max_objs]

            def masked_softmax(scores, mask):
                # subtract max for stability
                scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
                # compute padded softmax
                exp_scores = tf.exp(scores)
                exp_scores = exp_scores*mask + 1e-10
                exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
                return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])
            #self.obj_mask shape = [mini_batch_size, max_objs]
            self.softmax = masked_softmax(scores, self.obj_mask)
            #self.softmax shape = [mini_batch_size, max_objs]
            #self.targets (mini_batch_size,)
            self.selected_object = tf.argmax(self.softmax, axis=1)#row
            #self.selected_object shape = [mini_batch_size,]

            self.cross_entropy_loss = utils.cross_entropy(self.softmax, self.targets)
            self.loss = tf.reduce_mean(self.cross_entropy_loss)
            self.error = tf.reduce_mean(utils.error(self.softmax, self.targets))
            if policy_gradient:
                with tf.variable_scope('policy_gradient_loss'):
                    self.log_of_policy = tf.identity(self.cross_entropy_loss)
                    self.score_function = tf.multiply(self.log_of_policy, self.cum_rewards)
                    self.policy_gradient_loss = tf.reduce_mean(self.score_function)
                    self.loss = self.policy_gradient_loss

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

    """
    MemoryAttention approach followed the two papers:
    IJCAI2018, Rui Zhao and Volker Tresp, <Improving Goal-Oriented Visual Dialog Agents via Advanced Recurrent Nets with Tempered Policy Gradient>
    CVPR2019,  Ehsan Abbasnejad and Qi Wu and Javen Shi and Anton van den Hengel, <What’s to know? Uncertainty as a Guide to Asking Goal-oriented Questions>
    """
    def Memory(self):
        """
        memory pool for question-answer pair:
        Memory：5 facts，f1=LSTM(q1+a1), f2=LSTM(q2+a2),f3=LSTM(q3+a3),...,f5=LSTM(q5+a5)
        """
        batch_size = tf.shape(self.dialog_3d_trans)[1]#self.config['batch_size']
        max_qlen = tf.shape(self.dialog_3d_trans)[-1]

        def cond(j,mlist,c,h):return tf.less(j,self.max_qnum)
        def body(j,mlist,c,h):
            q_j = self.dialog_3d_trans[j,:,:]#[max_qnum,batch_size,max_qlen] -> [batch_size,max_qlen]
            a_j = self.dialog_3d_answers_trans[j,:]#[max_qnum,batch_size] -> [batch_size,]
            qj_emb = utils.get_embedding(q_j,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)
            aj_emb = utils.get_embedding(a_j,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)
            qa_j = tf.concat([qj_emb,tf.tile(tf.expand_dims(aj_emb,axis=1),[1,max_qlen,1])],axis=2)#(batch_size, max_qlen,512+512)
            step_length = self.dialog_question_num[:,j]
            state_c,state_h = self.Encoder(c=c,h=h,x=qa_j,seq_len=step_length,reuse=True)
            mlist = tf.concat([mlist,tf.expand_dims(state_h,axis=0)],axis=0)
            return tf.add(j, 1),mlist,state_c,state_h

        init_prob_0 = self.obj_mask*tf.ones_like(self.obj_mask, dtype=tf.float32)
        init_prob = tf.div(init_prob_0,tf.reduce_sum(init_prob_0,axis=1,keepdims=True))#(batch_size, max_objs_num)
        #init_prob = tf.Print(init_prob,[init_prob,tf.rank(init_prob)],"Memory-init_prob: ")
        q1 = self.dialog_3d_trans[0,:,:]#[batch_size, max_qlen]
        a1 = self.dialog_3d_answers_trans[0,:]#[batch_size,]
        q1_emb = utils.get_embedding(q1,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=False)
        a1_emb = utils.get_embedding(a1,n_words=self.num_words,n_dim=self.config['word_emb_dim'],scope="input_word_embedding",reuse=True)
        #q1_emb (batch_size, max_qlen, 512), a1_emb (batch_size, 512)
        step_length = self.dialog_question_num[:,0]
        qa_1 = tf.concat([q1_emb,tf.tile(tf.expand_dims(a1_emb,axis=1),[1,max_qlen,1])],axis=2)#(batch_size, max_qlen,512+512)
        c,h = self.Encoder(c=self.decoder_zero_state_c,
                    h=self.decoder_zero_state_h,
                    x=qa_1,
                    seq_len=step_length,reuse=False)
        #c,h shape = [batch_size, 512]
        mlist = tf.expand_dims(h, axis=0)#(1,batch_size,512)

        j = tf.constant(1)
        _,self.facts,_,_ = tf.while_loop(cond, body,
                    [j,mlist,c,h],
                    shape_invariants=[\
                        j.get_shape(),\
                        tf.TensorShape([None,None,None]),\
                        c.get_shape(),\
                        h.get_shape()])
        #self.facts (max_qnum, batch_size, 512)

    def Attention(self,scope_name="GuesserAtten"):
        """
        Key : key1 = MLP( sum of the spatial and category embeddings of all objects )
        an attention mask: Attention1(f1) = hadamard-product(f1,key1)
        attended facts: sum of {Attention1(f1),Attention2(f2),...,Attention5(f5)}
        """
        """
        args:
            self.obj_embs, (batch_size, max_objs, 512)
            self.facts (max_qnum, batch_size, 512) (batch_size,300)
        """
        with tf.variable_scope(scope_name):
            batch_size = tf.shape(self.obj_embs)[0]
            key = tf.reduce_sum(self.obj_embs,axis=1)#(batch_size,512)
            key = utils.fully_connected(key,self.config['num_lstm_units'],reuse=False,scope='keyfc',activation='tanh')#(batch_size,300)
            #self.is_training 1 or 0
            #key = tf.cond(self.is_training,\
            #    lambda:tf.nn.dropout(key,keep_prob=self.config['keep_prob']),\
            #    lambda:key)#is effective to avoid overfitting, but not enough
            attended_facts = tf.zeros(shape=[batch_size,self.config['num_lstm_units']],dtype=tf.float32)
            def cond(j,attended_facts):return tf.less(j,self.max_qnum)
            def body(j,attended_facts):
                fact_j = self.facts[j,:,:]#(batch_size,300)
                attention = tf.multiply(fact_j,key)#(batch_size,300)
                attended_facts = tf.add(attended_facts,attention)
                return tf.add(j,1),attended_facts
            j = tf.constant(0)
            _,self.attended_fact = tf.while_loop(cond,body,[j,attended_facts],shape_invariants=[j.get_shape(),tf.TensorShape([None,None])])
            #self.attended_fact (batch_size,512)

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

