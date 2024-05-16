# import tensorflow as tf
#
# from generic.tf_models import utils
# from  generic.tf_utils.abstract_network import AbstractNetwork
#
#

from generic.tf_utils.evaluator import Evaluator

import numpy as np

class QGenSamplingWrapper(object):
    def __init__(self, qgen, tokenizer, max_length):

        self.qgen = qgen

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.evaluator = None

        # Track the hidden state of LSTM
        self.state_c = None
        self.state_h = None
        self.state_size = int(qgen.decoder_zero_state_c.get_shape()[1])
        #self.objects_num = int(qgen.objects_j.get_shape()[1])
        #self.object_feature_size = int(qgen.objects_j.get_shape()[2])
        self.objects_j = None
        self.v_j = None

    def initialize(self, sess):
        self.evaluator = Evaluator(self.qgen.get_sources(sess), self.qgen.scope_name)

    def reset(self, batch_size):
        # reset state
        self.state_c = np.zeros((batch_size, self.state_size))
        self.state_h = np.zeros((batch_size, self.state_size))
        
        #QGen-V3.0
        #self.objects_j = np.zeros((batch_size, self.objects_num, self.object_feature_size))
        #self.v_j = np.zeros((batch_size, self.object_feature_size))

    def sample_next_question(self, sess, prev_answers, game_data, greedy,prev_question,prev_qlen,q_no,prob,prev_answer,is_train):

        game_data["dialogues"] = prev_answers
        game_data["seq_length"] = [1]*len(prev_answers)
        game_data["state_c"] = self.state_c
        game_data["state_h"] = self.state_h
        game_data["greedy"] = greedy
        game_data["prev_question"] = prev_question
        game_data["prev_qlen"] = prev_qlen
        game_data["is_first_question"] = q_no==0
        #game_data["objects_j"] = self.objects_j
        #game_data["v_j"] = self.v_j
        game_data["prob"] = prob
        game_data["prev_answer"] = prev_answer
        game_data["is_training"] = is_train
       
        # sample
        res = self.evaluator.execute(sess, self.qgen.samples, game_data)
        
        self.state_c = res[0]
        self.state_h = res[1]
        transpose_questions = res[2]
        seq_length = res[3]
        #self.v_j = res[5]#[batch_size, object_feature_size]
        #self.objects_j = res[6]#[batch_size, objects_num, object_feature_size]
        prob_j = res[6]#[batch_size, objects_num]
        #consin_scalar = res[7] #[1,batch_size]

        # Get questions
        padded_questions = transpose_questions.transpose([1, 0])
        padded_questions = padded_questions[:,1:]  # ignore first token

        for i, l in enumerate(seq_length):padded_questions[i, l:] = self.tokenizer.padding_token
        questions = [q[:l] for q, l in zip(padded_questions, seq_length)]
        return padded_questions, questions, seq_length,prob_j
    
    def last_question_prob(self,sess, last_answer, game_data, prev_question,prev_qlen,q_no,prob):
        game_data["prev_question"] = prev_question
        game_data["prev_answer"] = last_answer
        game_data["prev_qlen"] = prev_qlen
        game_data["is_first_question"] = q_no==0
        #game_data["objects_j"] = self.objects_j
        #game_data["v_j"] = self.v_j
        game_data["prob"] = prob
        
        res = self.evaluator.execute(sess, self.qgen.guessing_prob, game_data)
        return res

