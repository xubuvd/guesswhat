import sys
import logging
import numpy as np
import collections
from tqdm import tqdm
from neural_toolbox.attention import compute_intrinsic_reward
from generic.data_provider.nlp_utils import padder,padder_dual,padder_3d
from guesswhat.models.looper.tools import clear_after_stop_dialogue, list_to_padded_tokens

np.set_printoptions(precision=4,threshold=np.inf)
np.set_printoptions(suppress=True)

class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):
        self.storage = []

        self.log_level = 0
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.config = config
        self.max_no_question = config['loop']['max_question']
        self.max_depth = config['loop']['max_depth']
        self.k_best = config['loop']['beam_k_best']
        self.tau = 1.0/config['loop']['tau']
        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper
        self.logger = logging.getLogger()
 
    def process(self, sess, iterator, mode, optimizer=list(), store_games=False,log_level=0,epoch=0,G_optimizer=list()):
        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)
        self.log_level = log_level
        batch_no = 0
        self.epoch = epoch
        self.storage = []
        score, total_elem = 0, 0
        data_print_id = 0
        for game_data in tqdm(iterator):
            batch_no += 1

            # initialize the dialogue
            full_dialogues = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_answers = full_dialogues
            answers = [np.array(self.tokenizer.start_token) for _ in range(self.batch_size)]
            prev_qa = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_qa, prev_qa_length = padder(prev_qa, padding_symbol=self.tokenizer.padding_token)
            answers_pos_dialog = [np.array([]) for _ in range(self.batch_size)]

            prob_objects = []
            #dialog_questions = [[] for _ in range(self.batch_size)]
            #dialog_answers = [[] for _ in range(self.batch_size)]
            #consin_batch = [np.array([0 for _ in range(self.batch_size)])]#(1,batch_size)
            no_elem = len(game_data["raw"])
            total_elem += no_elem

            prob_j = np.full((self.batch_size,self.config['loop']['object_num']), 1.0/float(self.config['loop']['object_num']), dtype=np.float32)
            #[batch_size, objects_num]
            prob_list = list()

            #print("=================batch_no:{}======================".format(batch_no))
            # Step 1: generate question/answer
            self.qgen.reset(batch_size=no_elem)
            for no_question in range(self.max_no_question):
                # Step 1.1: Generate new question
                padded_questions, questions, seq_length,prob_j = \
                    self.qgen.sample_next_question(sess, prev_answers, game_data=game_data, mode=mode,\
                            prev_question=prev_qa,\
                            prev_answer=answers,\
                            prev_qlen=prev_qa_length,\
                            q_no=no_question,\
                            prob=prob_j)
                #print("no_question:{}\nprob:{}".format(no_question,prob_j))
                #print("no_question:{}\nconsin_scalar:{}".format(no_question,consin_scalar))
                prob_list.append(prob_j)#include the first uniform distribution
                #self.logger.info("no_question:{}:\t{}".format(no_question,np.shape(padded_questions)))
                """
                [[   8  204  102   10   10  334  107   15   13   16   12    0]
                 [   8    9  109   12    0    0    0    0    0    0    0    0]
                 [  34   10   18  124   12    0    0    0    0    0    0    0]
                 [   3    0    0    0    0    0    0    0    0    0    0    0]] (64,12)
                """
                # Step 1.2: Answer the question
                answers = self.oracle.answer_question(sess,
                                                      question=padded_questions,
                                                      seq_length=seq_length,
                                                      game_data=game_data)
                """
                answers:[6, 5, 6, 5, 5, 6, 6, 5, 5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5,......] (64,)
                """
                # Step 1.3: store the full dialogues
                prev_qa = [np.array([]) for _ in range(self.batch_size)]
                for i in range(self.batch_size):
                    full_dialogues[i] = np.concatenate((full_dialogues[i], questions[i], [answers[i]]))
                    answers_pos_dialog[i] = np.concatenate((answers_pos_dialog[i],[len(full_dialogues[i])-1]))
                    prev_qa[i] = np.concatenate((prev_qa[i],questions[i]))
                #consin_batch = np.concatenate((consin_batch,consin_scalar),axis = 0)

                prev_qa, prev_qa_length = padder(prev_qa, padding_symbol=self.tokenizer.padding_token)
                # Step 1.4 set new input tokens
                prev_answers = [[a] for a in answers]

                # Step 1.5 Compute the probability of finding the object after each turn
                if store_games:
                    padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
                    _, softmax, _ = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
                    prob_objects.append(softmax)

                # Step 1.6 Check if all dialogues are stopped
                has_stop = True
                for i,d in enumerate(full_dialogues):
                    has_stop &= self.tokenizer.stop_dialogue in d
                if has_stop:
                    print("no_question:{} has_stop, batch_size:{}, batch_no:{}".format(no_question,len(full_dialogues),batch_no))
                    break

            # Step 2 : clear question after <stop_dialogue>
            full_dialogues,_ = clear_after_stop_dialogue(full_dialogues, self.tokenizer)
            dialog_questions,dialog_answers,last_question,last_answer,dialog_real_qnum = self.sample_question_answer_pairs(full_dialogues,answers_pos_dialog)
            padded_dialog_questions,num_qa_pairs,max_qnum = padder_dual(dialog_questions,padding_symbol=self.tokenizer.padding_token)
            padded_dialog_answers,answer_len = padder(dialog_answers, padding_symbol=self.tokenizer.padding_token)#[batch_size, max_qnum]

            last_question = padded_dialog_questions[:,-1,:]
            last_answer = padded_dialog_answers[:,-1]
            padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)

            padded_last_question,last_seq_len = padder(last_question, padding_symbol=self.tokenizer.padding_token)
            last_prob = self.qgen.last_question_prob(sess,\
                        last_answer,\
                        game_data=game_data,\
                        prev_question=padded_last_question,\
                        prev_qlen=last_seq_len,\
                        q_no=5,\
                        prob=prob_j)
            prob_list.append(last_prob)#if bs is set to 8, then shape of prob_list is (6,8,49)
            if self.log_level >= 1:# and data_print_id % 500 == 0:
                self.logger.info("Epoch:{}\tprob_list:\n{}\t{}".format(epoch,prob_list,np.shape(prob_list)))
 
            # Step 3 : Find the object; found, softmax, selected_object
            """
            #Original version of Guesser [Strub et al., IJCAI 2017]
            found_object, last_softmax, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            """
            found_object, last_softmax, id_guess_objects, predicted_sequence = self.guesser.find_object(\
                                sess=sess, dialogues=padded_dialogue,seq_length=seq_length, game_data=game_data,\
                                dialog_3d=padded_dialog_questions,\
                                dialog_3d_answer=padded_dialog_answers,\
                                dialog_question_num=num_qa_pairs,max_qnum=max_qnum)
            #print("found_object:{}".format(found_object))
            """
            found_object:[  False  True  True False  True  True False  True False False False False
                            True  True  True  True False False  True False  True False  True False
                            True False False False  True False  True False False False False  True
                            False  True  True False False False  True False False False False False
                            True False False  True  True False  True False  True False False  True
                            True False False False]
            """

            """
            if len(G_optimizer) > 0:
                guesser_found_object,guesser_full_dialogues,guesser_game_data,guesser_questions,guesser_answers = self.filter(found_object,\
                                        full_dialogues,game_data,\
                                        dialog_questions,\
                                        dialog_answers)
                guesser_padded_dialogue, guesser_seq_length = list_to_padded_tokens(guesser_full_dialogues, self.tokenizer)
                guesser_padded_dialog_questions,guesser_num_qa_pairs,guesser_max_qnum = padder_dual(guesser_questions,padding_symbol=self.tokenizer.padding_token)
                guesser_padded_dialog_answers,guesser_answer_len = padder(guesser_answers, padding_symbol=self.tokenizer.padding_token)#[batch_size, max_qnum]
            """

            if self.log_level >= 1 and data_print_id % 500 == 0:
                for i in range(self.batch_size):
                    url = "{}".format(game_data["raw"][i].image.id)
                    game = self.tokenizer.decode(full_dialogues[i])
                    guess = found_object[i]
                    self.logger.info("Epoch:{}\tlookBatch:{}\turl:{}\t{}\tGuesser: {}".format(epoch,i,url,game,guess))
            if self.log_level >= 3:# and data_print_id % 500 == 0:
                batch_rewards = list()#real_qnum = min(len(prob_list),max_qnum)
                for j in range(1,len(prob_list)):
                    reward_j = compute_intrinsic_reward(prob_list[j-1],prob_list[j])#[batch_size,] = reward of the (j-1)-th question
                    batch_rewards.append(reward_j)
                batch_rewards = np.array(batch_rewards)#(max_qnum,batch_size)
                #batch_rewards_normed = np.transpose(batch_rewards/batch_rewards.sum(axis=0,keepdims=True))#(batch_size,max_qnum)
                batch_rewards[batch_rewards<=0] = 0
                batch_rewards_power = np.power(batch_rewards, self.tau)
                batch_rewards_normed = np.transpose(batch_rewards_power/batch_rewards_power.sum(axis=0,keepdims=True))
                prob_list_tr = np.transpose(np.array(prob_list), (1, 0, 2))##(batch_size,max_qnum+1,objects_num)
                targets_guess_state = [np.array([]) for _ in range(self.batch_size)]
                row_num = np.shape(predicted_sequence)[0]
                col_num = np.shape(predicted_sequence)[1]
                for n in range(0,row_num,max_qnum):
                    i = int(n/max_qnum)
                    targets_guess_state[i] = []
                    for j in range(max_qnum):
                        k = n + j
                        targets_guess_state[i].append(predicted_sequence[k,target_pos])
                #dialog_questions, dialog_answers
                for i in range(self.batch_size):
                    prob_sort_idx = np.argsort(prob_list_tr[i,:,:], axis=1)#(max_qnum+1,objects_num)
                    url = "{}".format(game_data["raw"][i].image.id)
                    q_a_g = "" 
                    for j in range(len(dialog_questions[i])):
                        q = self.tokenizer.decode(dialog_questions[i][j])
                        a = self.tokenizer.decode([dialog_answers[i][j]])
                        ig = "{:.4f}".format(targets_guess_state[i][j])#(batch_rewards_normed[i,j])#index 5 is out of bounds for axis 0 with size 5
                        if q_a_g != "":q_a_g += "#"
                        ans = ""
                        for k in range(self.config['loop']['object_num']-1,self.config['loop']['object_num']-1-10,-1):
                            pos_object_max = prob_sort_idx[j+1, k]
                            val_pos_object_max = prob_list_tr[i, j+1, pos_object_max]
                            if ans != "":ans += " "
                            ans += "{}:{:.4f}".format(pos_object_max,val_pos_object_max)
                        q_a_g += q+"||"+a+"||"+str(ig)+"||"+ans
                    #epoch \t image_id \t [qa||a1||ig||save_objects_wei #] \t Guesser \t targets_index \t id_guess_objects \t prob_objects
                    target_object_id = game_data["raw"][i].objects[game_data["targets_index"][i]].id
                    guess_object_id = game_data["raw"][i].objects[id_guess_objects[i]].id
                    prob_object = np.max(last_softmax[i,:])
                    self.logger.info("{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}".format(epoch,url,q_a_g,found_object[i],target_object_id,guess_object_id,prob_object))
                    self.logger.info("guessing_state:\t{}\t{}\t{}".format(url,found_object[i],targets_guess_state[i]))
            data_print_id += 1 
            score += np.sum(found_object)
            if store_games:
                prob_objects = np.transpose(prob_objects, axes=[1,0,2])
                for i, (d, g, t, f, go, po) in enumerate(zip(full_dialogues, game_data["raw"], game_data["targets_index"], found_object, id_guess_objects, prob_objects)):
                    self.storage.append({"dialogue": d,
                                         "game": g,
                                         "object_id": g.objects[t].id,
                                         "success": f,
                                         "guess_object_id": g.objects[go].id,
                                         "prob_objects" : po} )
            if len(optimizer) > 0:
                final_reward = found_object + 0  # +1 if found otherwise 0
                self.apply_policy_gradient(sess,
                                final_reward=final_reward,
                                padded_dialogue=padded_dialogue,
                                seq_length=seq_length,
                                game_data=game_data,
                                optimizer=optimizer,
                                dialog_3d=padded_dialog_questions,
                                dialog_3d_len=num_qa_pairs,
                                max_qnum=max_qnum,
                                prob_list=prob_list,
                                dialog_answers=padded_dialog_answers,
                                dialog_Jnum=dialog_real_qnum)
            #Optimize Guesser using Policy Gradient Algorithm
            if len(G_optimizer) > 0: 
                final_reward = found_object + 0  # +1 if found otherwise 0
                #final_reward = np.array(guesser_found_object) + 0  #(batch_size,)
                #self.apply_guesser_policy_gradient(sess,guesser_padded_dialogue,guesser_seq_length,guesser_game_data,\
                #            final_reward,G_optimizer,guesser_padded_dialog_questions,guesser_padded_dialog_answers,guesser_num_qa_pairs,guesser_max_qnum)
                self.apply_guesser_policy_gradient(sess,padded_dialogue,seq_length,game_data,final_reward,G_optimizer,padded_dialog_questions,padded_dialog_answers,num_qa_pairs,max_qnum)
        score = 1.0 * score / iterator.n_examples
        return score

    def get_storage(self):
        return self.storage
    def compute_reward(self,final_reward,padded_dialogue,seq_length,prob_list,dialog_3d_len,max_qnum,dialog_Jnum):
        """
        args:
            final_reward,    [batch_size,]
            seq_length,      [batch_size,]
            padded_dialogue, [batch_size, max_seq_len]
            prob_list, list of [max_qnum+1, batch_size, objects_num]
            dialog_Jnum, real qnum in each game, [batch_size,]
        returns:
            cum_rewards, [batch_size, max_seq_len]
        """

        """the extrinsic reward"""
        extrinsic_reward = np.zeros_like(padded_dialogue, dtype=np.float32)#(batch_size,max_seq_len)
        for i, (end_of_dialogue, r, J) in enumerate(zip(seq_length, final_reward,dialog_Jnum)):
            reward = r
            extrinsic_reward[i, :(end_of_dialogue - 1)] = reward  # gamma = 1
        """
        cum_rewards:[[1. 1. 1. 0.]
        [1. 1. 0. 0.]]
        """
        return extrinsic_reward 
        """the intrinsic reward"""
        kpos = np.zeros((self.batch_size,),dtype=np.int32)
        intrinsic_reward = np.zeros_like(padded_dialogue, dtype=np.float32)
        if( max_qnum+1 != len(prob_list) or max_qnum > self.max_no_question ):
            print("qnumError:prob_list:{}\tmax_qnum:{}\tdialog_3d_len:{}\tfinal_reward:{}\tseq_length:{}".format(\
                len(prob_list),max_qnum,np.shape(dialog_3d_len),np.shape(final_reward),np.shape(seq_length)))
        batch_rewards = list()
        #real_qnum = min(len(prob_list),max_qnum)
        for j in range(1,len(prob_list)):
            reward_j = compute_intrinsic_reward(prob_list[j-1],prob_list[j])#[batch_size,] = reward of the (j-1)-th question
            batch_rewards.append(reward_j)
        #batch_rewards, [real_qnum, batch_size]
        batch_rewards = np.array(batch_rewards)

        #Intrinsic reward: old 
        #batch_rewards_normed = batch_rewards/batch_rewards.sum(axis=0,keepdims=True)#(max_qnum, batch_size)/(1,batch_size) -> (max_qnum, batch_size)
        batch_rewards[batch_rewards<=0] = 0
        batch_rewards_power = np.power(batch_rewards, self.tau)
        batch_rewards_normed = batch_rewards_power/batch_rewards_power.sum(axis=0,keepdims=True)

        #print("real_qnum:{}\t batch_rewards_normed:{}\t intrinsic_reward:{}".format(real_qnum,np.shape(batch_rewards_normed),np.shape(intrinsic_reward)))
        #print("dialog_3d_len:{}".format(np.shape(dialog_3d_len)))
        for j in range(0,max_qnum):
            for i in range(self.batch_size):
                start = kpos[i]
                end = kpos[i] + dialog_3d_len[i,j]#index 3 is out of bounds for axis 1 with size 3
                if final_reward[i] == 1:
                    #print("i:{}\t j:{}\t start:{}\t end:{}".format(i,j,start,end))
                    intrinsic_reward[i,start:end] = batch_rewards_normed[j,i]#IndexError: index 4 is out of bounds for axis 0 with size 4
                kpos[i] += dialog_3d_len[i,j]
        cum_rewards = intrinsic_reward + extrinsic_reward
        if self.log_level >= 1:
            self.logger.info("Epoch:{}\tJnum:\n{}\t{}".format(self.epoch,dialog_Jnum,np.shape(dialog_Jnum)))
            self.logger.info("Epoch:{}\tbatch_rewards_normed:\n{}\t{}".format(self.epoch,batch_rewards_normed,np.shape(batch_rewards_normed)))
            self.logger.info("Epoch:{}\tintrinsic_reward:\n{}\t{}".format(self.epoch,intrinsic_reward,np.shape(intrinsic_reward)))
            self.logger.info("Epoch:{}\textrinsic_reward:\n{}\t{}".format(self.epoch,extrinsic_reward,np.shape(extrinsic_reward)))
            self.logger.info("Epoch:{}\tcum_rewards:\n{}\t{}".format(self.epoch,cum_rewards,np.shape(cum_rewards)))
        return cum_rewards

    def apply_policy_gradient(self, sess, final_reward, padded_dialogue, seq_length, game_data,\
                                optimizer,dialog_3d,dialog_3d_len,max_qnum,prob_list,dialog_answers,dialog_Jnum):

        # Compute cumulative reward TODO: move into an external function
        #cum_rewards = np.zeros_like(padded_dialogue, dtype=np.float32)
        #for i, (end_of_dialogue, r) in enumerate(zip(seq_length, final_reward)):
        #    cum_rewards[i, :(end_of_dialogue - 1)] = r  # gamma = 1

        cum_rewards = self.compute_reward(final_reward,padded_dialogue,seq_length,prob_list,dialog_3d_len,max_qnum,dialog_Jnum)

        # Create answer mask to ignore the reward for yes/no tokens
        answer_mask = np.ones_like(padded_dialogue)  # quick and dirty mask -> TODO to improve
        answer_mask[padded_dialogue == self.tokenizer.yes_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.no_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.non_applicable_token] = 0

        # Create padding mask to ignore the reward after <stop_dialogue>
        padding_mask = np.ones_like(padded_dialogue)
        padding_mask[padded_dialogue == self.tokenizer.padding_token] = 0
        # for i in range(np.max(seq_length)): print(cum_rewards[0][i], answer_mask[0][i],self.tokenizer.decode([padded_dialogue[0][i]]))

        # Step 4.4: optim step
        qgen = self.qgen.qgen  # retrieve qgen from wrapper (dirty)

        #print("sessrun_seq_length:{}".format(seq_length))
        #sessrun_seq_length:[48, 44, 48, 27, 38, 43, 43, 28, 35, 24, 37, 30, 28, 36, 28, 30, 37, 36, 37, 35, 43, 33, 23, 44, 25, 30, 33, 42, 36, 37, 48, 32, 35, 46, 41, 42, 34, 33, 38, 36, 27, 44, 29, 33, 38, 21, 29, 29, 39, 51, 38, 40, 43, 49, 26, 34, 32, 28, 54, 27, 32, 40, 40, 26]

        # Optimize QGen using Policy Gradient Algorithm
        sess.run(optimizer,
            feed_dict={
                'qgen/images:0':game_data["images"],
                'qgen/dialogues:0':padded_dialogue,
                'qgen/seq_length:0':seq_length,
                'qgen/padding_mask:0':padding_mask,
                'qgen/answer_mask:0':answer_mask,
                'qgen/cum_reward:0':cum_rewards,
                'qgen/dialog_3d:0':dialog_3d,
                'qgen/dialog_question_num:0':dialog_3d_len,
                'qgen/max_qnum:0':max_qnum,
                'qgen/dialog_3d_answer:0':dialog_answers
            })

    def sample_question_answer_pairs(self,full_dialogues,answers_pos_dialog):
        #guesser_dialog_questions = [np.array([]) for _ in range(self.batch_size)]
        dialog_questions = [np.array([]) for _ in range(self.batch_size)]
        dialog_answers = [np.array([]) for _ in range(self.batch_size)]
        last_question = [np.array([]) for _ in range(self.batch_size)]
        last_answer = []
        dialog_real_qnum = []
        for i in range(self.batch_size):
            start = 1#exclude the <start>
            dialogue = full_dialogues[i]
            dialog_questions[i] = []
            dialog_answers[i] = []
            last_question[i] = []
            prev_token = list([self.tokenizer.start_token])
            for idx,ans_pos in enumerate(answers_pos_dialog[i]):
                ans_pos = int(ans_pos)
                if ans_pos >= len(dialogue):break
                q = prev_token + list(dialogue[start:ans_pos])
                #guesser_dialog_questions[i].append(list(dialogue[start:ans_pos]))
                a = dialogue[ans_pos]
                start = ans_pos + 1
                dialog_questions[i].append(q)
                dialog_answers[i].append(a)
                prev_token = list([a])
            if len(dialog_questions[i]) > 0:
                last_question[i] = dialog_questions[i][-1][:]
                last_answer.append(dialog_answers[i][-1])
            else:
                last_question[i] = []
                last_answer.append(self.tokenizer.non_applicable_token)
            dialog_real_qnum.append(len(dialog_questions[i]))
        return dialog_questions,dialog_answers,last_question,last_answer,dialog_real_qnum
    def apply_guesser_policy_gradient(self,sess,padded_dialogue,seq_length,game_data,final_reward,G_optimizer,questions,answers,guesser_num_qa_pairs,guesser_max_qnum):
        guesser = self.guesser.guesser
        sess.run(G_optimizer,feed_dict={
                #'guesser/dialogues:0':padded_dialogue,
                #'guesser/seq_length:0':seq_length,
                'guesser/obj_mask:0':game_data["obj_mask"],
                'guesser/obj_cats:0':game_data["obj_cats"],
                'guesser/obj_spats:0':game_data["obj_spats"],
                'guesser/targets_index:0':game_data["targets_index"],
                'guesser/cum_rewards:0':final_reward,
                'guesser/dialog_3d:0':questions,
                'guesser/dialog_3d_answer:0':answers,
                'guesser/dialog_question_num:0':guesser_num_qa_pairs,
                'guesser/max_qnum:0':guesser_max_qnum})
    def filter(self,found_object,full_dialogues,game_data,dialog_questions,dialog_answers):
        """
        selecting the successed dialogues.
        args:
            found_object, (batch_size,)
            full_dialogues, list of list, (batch_size, ?)
            game_data,
        returns:
            dialogues, list of list, (size of successed cases,)
            games,
        """
        dialogues = []
        questions = []
        answers = []
        successed_found = []
        batch = collections.defaultdict(list)
        for i in range(self.batch_size):
            if found_object[i]:
                successed_found.append(found_object[i])
                dialogues.append(full_dialogues[i])
                questions.append(dialog_questions[i])
                answers.append(dialog_answers[i])
                batch['raw'].append(game_data["raw"][i])
                batch['obj_spats'].append(game_data["obj_spats"][i])
                batch['obj_cats'].append(game_data["obj_cats"][i])
                batch['targets_index'].append(game_data["targets_index"][i])
                batch['targets_spatial'].append(game_data["targets_spatial"][i])
                batch['targets_category'].append(game_data["targets_category"][i])
                batch['debug'].append(game_data["debug"][i])
                batch["images"].append(game_data["images"][i])
        # Pad objects
        batch['obj_spats'], obj_length = padder_3d(batch['obj_spats'])
        batch['obj_cats'], obj_length = padder(batch['obj_cats'])

        #Compute the object mask
        max_objects = max(obj_length)
        batch_size = len(questions)
        batch['obj_mask'] = np.zeros((batch_size, max_objects), dtype=np.float32)
        for i in range(batch_size):batch['obj_mask'][i, :obj_length[i]] = 1.0
        return successed_found,dialogues,batch,questions,answers

