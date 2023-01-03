""" import standard library """
import itertools
import os
import logging as log
import argparse
import time
import _pickle as pkl
from typing import Any, Dict, Iterable, List, Sequence, Type, Union
import pyhocon
import types
import sys
from tqdm import tqdm
import copy
from collections import defaultdict, OrderedDict
import re
import random
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn import (
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch.nn import LayerNorm, MultiheadAttention, Linear, Dropout
from allennlp.nn.util import move_to_device, device_mapping
from allennlp.data import Instance, Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import (
    TextField,
    LabelField,
    ListField,
    MetadataField,
    MultiLabelField,
    SpanField,
)
import logging as log

from dvq_model.task import pad_idx, eos_idx

""" utility function """


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_device = 0
    else:
        device = torch.device("cpu")
        cuda_device = -1
    return device, cuda_device


def input_from_batch(batch,batch_dvq2topic_ids):
    
    assert batch.size() == batch_dvq2topic_ids.size()
    src_nopad_mask = batch != 0 
    nopad_lengths = torch.max(src_nopad_mask.sum(dim=-1).int()).item()
    sent = batch[:,0:nopad_lengths].clone() 
    batch_dvq2topic_ids = batch_dvq2topic_ids[:,0:nopad_lengths].clone() 
    batch_size, seq_len = sent.size()
    # no <SOS> and <EOS>
    enc_in = sent[:, 1:-1].clone()
    batch_dvq2topic_ids = batch_dvq2topic_ids[:,1:-1].clone()
    enc_in[enc_in == eos_idx] = pad_idx  

    # no <SOS>
    dec_out_gold = sent[:, 1:].contiguous()

    # no <EOS>
    dec_in = sent[:, :-1].clone()
    dec_in[dec_in == eos_idx] = pad_idx

    assert enc_in.size() == batch_dvq2topic_ids.size()
    out = {
        "batch_size": batch_size,
        "dec_in": dec_in,
        "dec_out_gold": dec_out_gold,
        "enc_in": enc_in,
        "sent": sent,
        
        "batch_dvq2topic_ids":batch_dvq2topic_ids,
    }
    return out

def batched_index_select(input, dim, index):
    views = [1 if i != dim else -1 for i in range(len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index_ = index.view(views).expand(expanse)
    output = torch.gather(input, dim, index_).view(index.size()[0],index.size()[1],input.size()[1])
    return output



def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PackedSequneceUtil(object):
    def __init__(self):
        self.is_packed = False
        self.pack_shape = None

    def preprocess(self, input):
        self.is_packed = isinstance(input, PackedSequence)
        if self.is_packed:
            input, *self.pack_shape = input
        return input

    def postprocess(self, output, pad):
        assert self.is_packed
        packed_ouput = PackedSequence(output, *self.pack_shape)
        padded_output = pad_packed_sequence(
            packed_ouput, batch_first=True, padding_value=pad
        )[0]
        return padded_output

