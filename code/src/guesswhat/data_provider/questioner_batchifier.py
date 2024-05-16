import numpy as np
import itertools
import collections
import copy
from tqdm import tqdm
from generic.data_provider.batchifier import AbstractBatchifier
from generic.data_provider.image_preprocessors import (resize_image, get_spatial_feat,scaled_crop_and_pad)
from generic.data_provider.nlp_utils import padder, padder_3d, padder_dual

answer_dict = \
    {'Yes': np.array([1, 0, 0], dtype=np.int32),
    'No': np.array([0, 1, 0], dtype=np.int32),
    'N/A': np.array([0, 0, 1], dtype=np.int32)
    }

class QuestionerBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(),flag=True, **kwargs):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.kwargs = kwargs
        self.flag = flag

    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games
    
    #Data Augmentation
    def permutation(self,games,augmented_factor=1):
        number_games = len(games)
        for idx in tqdm(range(number_games)):
            for k in range(augmented_factor):
                rseed = np.random.randint(0,1000000)
                np.random.seed(rseed)
                g_augmented = copy.deepcopy(games[idx])
                state = np.random.get_state()
                np.random.shuffle(g_augmented.question_ids)
                np.random.set_state(state)
                np.random.shuffle(g_augmented.questions)
                np.random.set_state(state)
                np.random.shuffle(g_augmented.answers)
                games.append(g_augmented)

    def apply(self, games):
        """
        games = list of game
        """
        batch = collections.defaultdict(list)
        batch_size = len(games)

        all_answer_indices = []
        for i, game in enumerate(games):
            #batch['raw'].append(game)
            """
            Flattened question answers
            """
            round_count = 0
            q_tokens = [self.tokenizer.apply(q) for q in game.questions]
            a_tokens = [self.tokenizer.apply(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add start token
            qa_tokens = []
            one_batch = list()
            one_batch_answer = list() 
            answer_indices = []
            cur_index = 0
            prev_token = [self.tokenizer.start_token]
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok

                qa_tokens += prev_token
                qa_tokens += q_tok
                # Compute index of answer in the full dialogue
                answer_indices += [cur_index + len(q_tok) + 1]
                cur_index = answer_indices[-1]
                
                one_batch.append(qa_tokens)
                one_batch_answer.append(a_tok[0])
                qa_tokens = []
                prev_token = a_tok
                round_count += 1
                if round_count > 10:break
            tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogues"].append(tokens)
            batch["dialog_3d"].append(one_batch)#[batch_size, question_num, question_len]
            batch["dialog_3d_answer"].append(one_batch_answer)#[batch_size, question_num]
            all_answer_indices.append(answer_indices)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):
                spatial = get_spatial_feat(obj.bbox, game.image.width, game.image.height)
                category = obj.category_id

                if obj.id == game.object_id:
                    batch['targets_category'].append(category)
                    batch['targets_spatial'].append(spatial)
                    batch['targets_index'].append(index)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spats'].append(obj_spats)
            batch['obj_cats'].append(obj_cats)

            # image
            img = game.image.get_image()#vgg16_pool5 = (7,7,512)
            if img is not None:
                if "images" not in batch:  # initialize an empty array for better memory consumption
                    batch["images"] = np.zeros((batch_size,) + img.shape)
                    """
                    shape = (batch_size, fc8_size)
                    shape = (batch_size, 7, 7, 512)
                    """
                batch["images"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogues'], batch['seq_length'] = padder(batch['dialogues'], padding_symbol=self.tokenizer.padding_token)
        seq_length = batch['seq_length']
        max_length = max(seq_length)

        # Compute the token mask
        batch['padding_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['padding_mask'][i, (seq_length[i] + 1):] = 0.

        # Compute the answer mask
        batch['answer_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['answer_mask'][i, all_answer_indices[i]] = 0.

        # Pad objects
        batch['obj_spats'], obj_length = padder_3d(batch['obj_spats'])
        batch['obj_cats'], obj_length = padder(batch['obj_cats'])

        # Pad question number and question len
        #[batch_size, max_question_num, max_question_len]
        batch["dialog_3d"], batch["dialog_question_num"],batch["max_qnum"] = padder_dual(batch["dialog_3d"], padding_symbol=self.tokenizer.padding_token)
        """
        batch[dialog_question_num]:
           [[ 8  9  0  0  0]
            [ 6  0  0  0  0]
            [ 7  9  0  0  0]
            [ 7  5  7  5  7]
            [ 7  6  6  5  0]
            [ 6  5  6 11  5]
            [ 7  7  8  7  0]
            [ 7  6  7  5  5]
            [..............]]
        """
        batch["dialog_3d_answer"],_ = padder(batch["dialog_3d_answer"], padding_symbol=self.tokenizer.padding_token)#[batch_size, max_qnum]
        #print("batch[seq_length]:\n{}".format(batch['seq_length']))
        """
        batch[seq_length]:
        [26 22 28 31 38 18 14 16 37 14 30 32 35 28 36 7 20 30 14  8 32 15 33 23 14 20 31 21 17 23 14 19]
        """
        # Compute the object mask
        max_objects = max(obj_length)
        batch['obj_mask'] = np.zeros((batch_size, max_objects), dtype=np.float32)
        for i in range(batch_size):
            batch['obj_mask'][i, :obj_length[i]] = 1.0
        return batch

