from nltk.tokenize import TweetTokenizer
import json



"""
remove the following specical flags except <stop_dialogue> in action space:
"<padding>": 0, "<start>": 1, "<stop>": 2, "<stop_dialogue>": 3, "<unk>": 4, "<yes>": 5, "<no>": 6, "<n/a>": 7
"""

class GWTokenizer:
    """ """
    def __init__(self, dictionary_file):
        with open(dictionary_file, 'r') as f:
            self.word2i = json.load(f)['word2i']
        self.wpt = TweetTokenizer(preserve_case=False)

        if "<stop_dialogue>" not in self.word2i:
            self.word2i["<stop_dialogue>"] = len(self.word2i)

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.start_token = self.word2i["<start>"]
        self.stop_token = self.word2i["?"]
        self.stop_dialogue = self.word2i["<stop_dialogue>"]
        self.padding_token = self.word2i["<padding>"]
        self.yes_token = self.word2i["<yes>"]
        self.no_token = self.word2i["<no>"]
        self.non_applicable_token = self.word2i["<n/a>"]
        #"<yes>": 5, "<no>": 6, "<n/a>": 7

        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]
        self.inverse_answer = dict()
        self.inverse_answer[self.yes_token] = self.no_token
        self.inverse_answer[self.no_token] = self.yes_token
        self.inverse_answer[self.non_applicable_token] = self.non_applicable_token
    def inverse(self,ans):
        return self.inverse_answer[ans]

    """
    Input: String
    Output: List of tokens
    """
    def apply(self, question, is_answer=False):

        tokens = []
        if is_answer:
            token = '<' + question.lower() + '>'
            tokens.append(self.word2i[token])
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                tokens.append(self.word2i[token])
        return tokens

    def decode(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def split_questions(self, dialogue_tokens):
        qas = []
        qa = []
        for token in dialogue_tokens:
            assert token != self.padding_token, "Unexpected padding token"
            # check for end of dialogues
            if token == self.stop_dialogue:
                break
            if token == self.start_token:
                continue
            qa.append(token)
            # check for end of question
            if token in self.answers:
                qas.append(qa)
                qa = []
        return qas

