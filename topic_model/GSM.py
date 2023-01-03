import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from VAE import VAE
import matplotlib.pyplot as plt
import sys
import codecs
import time
sys.path.append('..')
from utils import evaluate_topic_quality, smooth_curve

class GSM:
    def __init__(self,bow_dim=10000,n_topic=20,taskname=None,device=None):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        #TBD_fc1
        self.vae = VAE(voc_size=bow_dim, n_topic=n_topic, dropout=0.0)
        self.device = device
        self.id2token = None
        self.taskname = taskname
        if device!=None:
            self.vae = self.vae.to(device)

    def train(self,train_data,batch_size=256,learning_rate=1e-3,test_data=None,num_epochs=100,is_evaluate=False,log_every=5,beta=1.0,criterion='cross_entropy',ckpt=None):
        self.vae.train()
        self.id2token = {v:k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=train_data.collate_fn)
        optimizer = torch.optim.Adam(self.vae.parameters(),lr=learning_rate)

        if ckpt:
            self.load_model(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst = []
        c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
        best_ppx = 9999
        for epoch in range(start_epoch, num_epochs):
            epochloss_lst = []
            ppx_sum = 0.0
            doc_count = 0.0
            for iter,data in enumerate(data_loader):
                word_count_list= []
                optimizer.zero_grad()
                txts,bows = data
                bows = bows.to(self.device)
                bows_recon,mus,log_vars, _, _, _ = self.vae(bows)
                logsoftmax = torch.log_softmax(bows_recon+1e-10,dim=1)
                rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
                loss = rec_loss + kl_div * beta
                loss.backward()
                optimizer.step()
                rec_loss_per = -1.0 * torch.sum(bows*logsoftmax, dim=1)
                rec_loss_per = rec_loss_per.cpu().detach().numpy()
                for bow in bows:
                    word_count = torch.sum(bow).item()
                    word_count_list.append(word_count)
                    
                word_count_np = np.array(word_count_list) 
                ppx_sum += np.sum(np.true_divide(rec_loss_per,word_count_np))
                doc_count += len(txts)

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))

            if (epoch+1) % log_every==0:
                save_name = f'./ckpt/topic_ckpt/{self.taskname}_{self.n_topic}.ckpt'
                topic_embedding_save_name = f'./ckpt/topic_ckpt/{self.taskname}_topic_embedding.ckpt'
                checkpoint = {
                    "net": self.vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                        "taskname": self.taskname
                    }
                }
                # The code lines between this and the next comment lines are duplicated with WLDA.py, consider to simpify them.
                train_ppx = np.exp(ppx_sum / doc_count)
                topic_words = self.show_topic_words()
                if test_data!=None:
                    text_ppx = self.evaluate(test_data,topic_words,calc4each=False)
                if best_ppx > text_ppx:
                    torch.save(checkpoint,save_name)
                    torch.save(self.vae.t.weight,topic_embedding_save_name)
                    #print("This is best test ppx!")
                    best_ppx = text_ppx
                print(f'taskname {self.taskname}, n_topic {self.n_topic} Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)},train_ppx:{train_ppx},text_PPX:{text_ppx}\t')
    
    def evaluate(self,test_data,topic_words,calc4each=False):
        self.vae.eval()
        data_loader = DataLoader(test_data,batch_size=32,shuffle=False,num_workers=4,collate_fn=test_data.collate_fn)
        ppx_sum = 0.0
        doc_count = 0.0
        for iter,data in enumerate(data_loader):
            word_count_list= []
            txts,bows = data
            bows = bows.to(self.device)
            bows_recon,mus,log_vars,_,_,_ = self.vae(bows)
            logsoftmax = torch.log_softmax(bows_recon+1e-10,dim=1)
            rec_loss = -1.0 * torch.sum(bows*logsoftmax)
            kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
            rec_loss_per = -1.0 * torch.sum(bows*logsoftmax, dim=1)
            rec_loss_per = rec_loss_per.cpu().detach().numpy()
            for bow in bows:
                word_count = torch.sum(bow).item()
                word_count_list.append(word_count)
                
            word_count_np = np.array(word_count_list) 
            ppx_sum += np.sum(np.true_divide(rec_loss_per,word_count_np))
            doc_count += len(txts)        
        ppx = np.exp(ppx_sum / doc_count)
        return ppx
        

    def show_topic_words(self,topic_id=None,topK=15, dictionary=None):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        word_dist = self.vae.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        vals,indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if self.id2token==None and dictionary!=None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id==None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
        return topic_words

    def load_model(self, model):
        self.vae.load_state_dict(model)


