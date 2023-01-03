import os
import re
import torch
import pickle
import argparse
import logging
from GSM import GSM 
from utils import *
from dataset import TrainData,TestData
from multiprocessing import cpu_count

parser = argparse.ArgumentParser('GSM topic model')
parser.add_argument('--taskname',type=str,default='stsb')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--n_topic',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=512)

args = parser.parse_args()

def main():

    device = torch.device('cuda')
    trainData = TrainData(args.taskname)
    testData = TestData(args.taskname,dictionary=trainData.dictionary)

    model = GSM(bow_dim=trainData.vocabsize, n_topic=args.n_topic,taskname=args.taskname,device=device)
    model.train(train_data=trainData,batch_size=args.batch_size,test_data=testData,num_epochs=args.num_epochs,log_every=10,beta=1.0)

if __name__ == "__main__":
    main()
