import numpy as np
from generic.utils.file_handlers import pickle_loader


class GloveEmbeddings(object):

    def __init__(self, file, glove_dim=300):
        self.glove = pickle_loader(file)
        self.glove_dim = glove_dim

    def get_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")
            if token in self.glove:
                vectors.append(np.array(self.glove[token]))
            else:
                vectors.append(np.zeros((self.glove_dim,)))
        return vectors

##[batch_size, question_num, question_len]
def padder_dual(list_of_tokens,q_num=None, padding_symbol=0, q_length=None):
    if q_num is None:
        q_num = np.array([len(q) for q in list_of_tokens])
    max_qnum = q_num.max()

    max_qlen = -1
    if q_length is None:
        for game in list_of_tokens:
            q_lens = np.array([len(q) for q in game])
            if len(q_lens) > 0 and q_lens.max() > max_qlen:max_qlen = q_lens.max()
    else:max_qlen = q_length
    
    batch_size = len(list_of_tokens)
    padded_tokens = np.full(shape=(batch_size, max_qnum, max_qlen), fill_value=padding_symbol)
    padded_question_nums = np.full(shape=(batch_size, max_qnum), fill_value=0)
    
    for batch_id, game in enumerate(list_of_tokens):
        for q_id, q in enumerate(game):
            #print("batch_id:{}\tq_id:{}".format(batch_id,q_id))
            padded_tokens[batch_id,q_id,:len(q)] = q[:max_qlen]
            padded_question_nums[batch_id,q_id] = len(q)# + 1 #why add 1, as including the <start> or answer position.
    return padded_tokens,padded_question_nums,max_qnum

def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

    if seq_length is None:
        seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)#[batch_size, 1]

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)

    padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq)] = seq

    return padded_tokens, seq_length

def padder_3d(list_of_tokens, max_seq_length=0):
    seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    feature_size = list_of_tokens[0][0].shape[0]

    padded_tokens = np.zeros(shape=(batch_size, max_seq_length, feature_size))

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq), :] = seq

    return padded_tokens, max_seq_length


class DummyTokenizer(object):
    def __init__(self):
        self.padding_token = 0
        self.dummy_list = list()
        self.no_words = 10
        self.no_answers = 10
        self.unknown_answer = 0

    def encode_question(self, _):
        return self.dummy_list

    def encode_answer(self, _):
        return self.dummy_list
