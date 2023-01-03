import os
import time
import numpy as np
import pandas as pd
import gensim
import pickle
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from collections import Counter
import sys
import csv

class TrainData(Dataset):
    def __init__(self,taskname):
        cwd = os.getcwd()
        csvPath = os.path.join(cwd,'data/',taskname, 'train.csv') 
        tmpDir = os.path.join(cwd,'data/',taskname,taskname)
        self.txtLines = []
        self.dictionary = None
        self.bows,self.docs = None,[]
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
        if os.path.exists(os.path.join(tmpDir,'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmpDir,'corpus.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmpDir,'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmpDir,'docs.pkl'),'rb'))
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()} 
        else:
            csv_reader = csv.reader(open(csvPath))
            for line_ in csv_reader:
                self.txtLines.append(line_[1])
                self.txtLines.append(line_[2])
            for doc in self.txtLines:
                self.docs.append(list(gensim.utils.tokenize(doc, lower=True)))
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_tokens(list(map(self.dictionary.token2id.get, STOPWORDS)))
            len_1_words = list(filter(lambda w: len(w) == 1, self.dictionary.values()))
            self.dictionary.filter_tokens(list(map(self.dictionary.token2id.get, len_1_words)))
            self.dictionary.filter_n_most_frequent(remove_n=20)  
            self.dictionary.filter_extremes(no_below=3, keep_n=None) 
            self.dictionary.token2id["UNK"] = len(self.dictionary)
            self.dictionary.compactify()
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()} 

            # convert to BOW representation
            self.bows, _docs = [],[]
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow!=[]:
                    _docs.append(list(doc))
                    self.bows.append(_bow)   
                else:
                    doc = ["UNK"]
                    _docs.append(list(doc))
                    _bow = self.dictionary.doc2bow(doc)
                    self.bows.append(_bow)                                    
            self.docs = _docs
            # serialize the dictionary
            gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir,'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmpDir,'dict.txt'))
            pickle.dump(self.docs,open(os.path.join(tmpDir,'docs.pkl'),'wb'))
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')
        
    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs
    
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

class TestData(Dataset):
    def __init__(self, taskname, dictionary=None, txtPath=None, tokenizer=None,stopwords=None,no_below=5,no_above=0.1):
        cwd = os.getcwd()

        csvPath = os.path.join(cwd,'data/',taskname, 'test.csv') 
        self.txtLines = []
        csv_reader = csv.reader(open(csvPath))
        for line_ in csv_reader:
            self.txtLines.append(line_[1])
            self.txtLines.append(line_[2])
        self.dictionary = dictionary
        self.bows,self.docs = [],[]

        print('Tokenizing ...')
        for doc in self.txtLines:
            self.docs.append(list(gensim.utils.tokenize(doc, lower=True)))
        # convert to BOW representation
        for doc in self.docs:
            if doc is not None:
                _bow = self.dictionary.doc2bow(doc)
                if _bow!=[]:
                    self.bows.append(_bow)
                else:
                    doc = ["UNK"]
                    _bow = self.dictionary.doc2bow(doc)
                    self.bows.append(_bow)          
            else:
                doc = ["UNK"]
                _bow = self.dictionary.doc2bow(doc)
                print("_bow",_bow.size())
                self.bows.append(_bow) 
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        assert  len(self.bows) == len(self.docs)
        print(f'Processed {len(self.bows)} documents.')

    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        item = list(zip(*self.bows[idx])) # item = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs

    def __iter__(self):
        for doc in self.docs:
            yield doc
            
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

class topicDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, test_data, task, topic_data_dir):
        super().__init__()
        self.dictionary = Dictionary.load_from_text(os.path.join(topic_data_dir+task,'dict.txt'))
        self.vocabsize = len(self.dictionary)
        
        self.txtlines = set()
        for line in test_data:
            text1 = line["sentence1"] 
            text2 = line["sentence2"] 
            self.txtlines.add(text1)
            self.txtlines.add(text2)
    
        self.test_docs = []
        for txt in self.txtlines:
            self.test_docs.append(list(gensim.utils.tokenize(txt, lower=True)))
    def text2bow(self, doc):
        _doc = list(gensim.utils.tokenize(doc, lower=True))
        _bow = self.dictionary.doc2bow(_doc)
        if _bow == []:
            _bow = [(self.vocabsize-1,1)]
        bow = torch.zeros(self.vocabsize)
        item = list(zip(*_bow)) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        return bow

