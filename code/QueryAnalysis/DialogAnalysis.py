# -*- coding:utf-8 -*-
import copy
import codecs

class QuestionClassification(object):
    def __init__(self, keywords_file):
        self.attributes_dict = dict()
        self.entity_dict = dict()

        self.all_key_words = set()
        self.second2firstmap = dict()
        self.entity = set(["Super-category","object"])
        self.Load(keywords_file)
    def Load(self,keywords_file):
        with codecs.open(keywords_file,'rb','utf-8') as f: data = f.readlines()
        for line in data:
            tokens = line.strip().split('\t')
            if len(tokens) != 3:continue
            first_type = tokens[0].strip()
            second_type = tokens[1].strip()
            self.second2firstmap[second_type] = first_type

            self.all_key_words.add(first_type)
            self.all_key_words.add(second_type)

            sub_tokens = tokens[2].strip().split(',')
            for item in sub_tokens:
                it = item.strip().strip('‘').strip('’').strip()
                if len(it) < 1:continue
                self.all_key_words.add(it)
                if second_type in self.entity:
                    if it not in self.entity_dict:self.entity_dict[it] = list()
                    if second_type not in self.entity_dict[it]:self.entity_dict[it].append(second_type)
                else:
                    if it not in self.attributes_dict:self.attributes_dict[it] = list()
                    if second_type not in self.attributes_dict[it]:self.attributes_dict[it].append(second_type)
    def Print(self):
        print("Attributes:{}".format(self.attributes_dict))
        print("Entity:{}".format(self.entity_dict))
        print("Map:{}".format(self.second2firstmap))

    def classify(self,query):
        qtype = list()
        tokens = query.strip().split(' ')
        for word in tokens:
            if word not in self.all_key_words:continue
            if word in self.attributes_dict:qtype += self.attributes_dict[word]
        if len(qtype) < 1:
            for word in tokens:
                if word not in self.all_key_words:continue
                if word in self.entity_dict:qtype += self.entity_dict[word]
        if len(qtype) < 1:qtype.append("not-classified")
        return list(set(qtype))

class DialogRepetition(object):
    def __init__(self):pass
    def check_repetition(self,dialogue):pass
                

if __name__ == "__main__":
    
    query_type = QuestionClassification("./keywords.list")
    query_type.Print()
    for line in open("./query.log"):
        query = line.strip().strip('?').strip('.').strip()
        type = query_type.classify(query)
        print("query:{}\ttype:{}".format(line.strip(),type))

