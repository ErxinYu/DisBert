import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import logging as log
import sys
from .vq_quantizer import DVQ
from .task import pad_idx, eos_idx
from .util import input_from_batch,batched_index_select,PositionalEncoding,generate_square_subsequent_mask



class TransformerQuantizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_embeddings = config.vocab_size

        self.embedding = nn.Embedding(
            config.vocab_size, config.transformer.d_model, padding_idx=0
        )
        self.pos_encoder = PositionalEncoding(
            d_model=config.transformer.d_model, dropout=config.transformer.dropout
        )
        self.topic_encoder = nn.Linear(config.transformer.d_model+config.quantizer.K, config.transformer.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer.d_model, #64
            nhead=config.transformer.nhead, #4 
            dim_feedforward=config.transformer.d_ffn, #256
            dropout=config.transformer.dropout, #0.2
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer.enc_nlayer #1
        )

        # sentence or word
        self.quantizer_level = config.quantizer.level
        # decompose
        split = config.quantizer.M
        D = config.transformer.d_model // split

        # specific to quantizer
        self.quantizer = DVQ(
            config,
            num_embeddings=config.quantizer.K, 
            embedding_dim=D, 
            split=split, 
            decompose_option="slice",
        )


        self.output_dim = config.transformer.d_model 
        self.beta = nn.Embedding(2000, config.transformer.d_model, padding_idx=0)
        

    def forward(self, src, topic_embedding, theta, beta, batch_dvq2topic_ids, pretrain_vq):
        src_pad_mask = src == 0 
        src_nopad_mask = src != 0 
        nopad_lengths = src_nopad_mask.sum(dim=-1).long() 
        src_emb = self.embedding(src).transpose(0, 1) 
        src_emb = self.pos_encoder(src_emb) 

        # topic-sense encoder
        t_i = batched_index_select(beta.transpose(0,1),0,batch_dvq2topic_ids) 
        theta = theta.unsqueeze(1) 
        te = torch.mul(t_i,theta) 
        src_emb = torch.cat((src_emb,te.transpose(0,1)),dim=2) 
        src_emb = self.topic_encoder(src_emb) 

        src_mask = None
        memory = self.encoder(
            src_emb, src_key_padding_mask=src_pad_mask, mask=src_mask
        ).transpose(0, 1) 
        packed_memory = pack_padded_sequence(
            memory, lengths=nopad_lengths.cpu(), batch_first=True, enforce_sorted=False
        ) 
        quantizer_out = self.quantizer(packed_memory, topic_embedding, theta, pretrain_vq)
        enc_out = quantizer_out["quantized"]


        return {
            "quantizer_out": quantizer_out,
            "nopad_mask": src_nopad_mask,
            "sequence": enc_out,
        }

    def get_output_dim(self):
        return self.output_dim


class DecodingUtil(object):
    def __init__(self, vsize):
        self.criterion = nn.NLLLoss(reduction="none") 
        self.vsize = vsize 
    def forward(self, logprobs, dec_out_gold): 
        loss_reconstruct = self.criterion(
            logprobs.contiguous().view(-1, self.vsize), dec_out_gold.view(-1)
        )
        # mask out padding
        nopad_mask = (dec_out_gold != pad_idx).view(-1).float() 
        nll = (loss_reconstruct * nopad_mask).view(logprobs.shape[:-1]).detach() 
        loss_reconstruct = (loss_reconstruct * nopad_mask).sum() 

        # post-processing
        nopad_mask2 = dec_out_gold != pad_idx 
        pred_idx = torch.argmax(logprobs, dim=2)
        pred_idx = pred_idx * nopad_mask2.long() 
        ntokens = nopad_mask.sum().item()

        return {
            "loss": loss_reconstruct,
            "pred_idx": pred_idx,
            "nopad_mask": nopad_mask2,
            "ntokens": ntokens,
            "nll": nll,
        }


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()

        self.encoder = encoder

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            dim_feedforward=config.transformer.d_ffn,
            dropout=config.transformer.dropout,
        )

        self.num_embeddings = config.vocab_size
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer.d_model, self.num_embeddings),
            nn.LogSoftmax(dim=-1),
        )

        self.decoding_util = DecodingUtil(config.vocab_size)
        self.init_weights()
        self.kl_fbp = config.concrete.kl.fbp_threshold
        self.kl_beta = config.concrete.kl.beta
        self.cludict = {}

    def forward(self, batch,pretrain_vq):
        input = input_from_batch(batch["input"],batch["batch_dvq2topic_ids"])
        topic_embedding = batch["topic_embedding"]
        theta = batch["theta"]
        beta = batch["beta"]
        batch_dvq2topic_ids = input["batch_dvq2topic_ids"]
        src = input["enc_in"] 
        bsz = src.shape[0]

        # encoder
        enc_outdict = self.encoder(src, topic_embedding, theta,beta, batch_dvq2topic_ids,pretrain_vq)
        indices = enc_outdict["quantizer_out"]["encoding_indices"]
        memory = enc_outdict["sequence"].transpose(0, 1) 
        return self.decode(input, memory, enc_outdict)

    def decode(self, input, memory, enc_outdict):

        # teacher forcing 
        dec_out_gold = input["dec_out_gold"]
        tgt = input["dec_in"]
        tgt_emb = self.encoder.embedding(tgt).transpose(0, 1)
        tgt_emb = self.encoder.pos_encoder(tgt_emb)
        bsz = input["enc_in"].shape[0]

        tgt_pad_mask = tgt == 0
        tgt_mask = generate_square_subsequent_mask(len(tgt_emb))
        tgt_mask = tgt_mask.to(tgt_pad_mask.device)
        src_nopad_mask = enc_outdict["nopad_mask"]
        src_pad_mask = src_nopad_mask == 0
        output = self.decoder_layer(
            tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )
        output = output.transpose(0, 1)
        logprobs = self.classifier(output)
        dec_outdict = self.decoding_util.forward(logprobs, dec_out_gold) 
        loss_reconstruct = dec_outdict["loss"]
        pred_idx = dec_outdict["pred_idx"]
        ntokens = dec_outdict["ntokens"]

        # total loss
        quantizer_out = enc_outdict["quantizer_out"]
        loss = loss_reconstruct + quantizer_out["loss"]
        result = {
            "loss_commit": quantizer_out["loss_commit"],
            "min_distances": quantizer_out["min_distances"],
        }

        result.update(
            {
                'z_q': quantizer_out['quantized'].detach(),
                "indices": quantizer_out["encoding_indices"].detach(),
                "loss_reconstruct": loss_reconstruct.detach(),
                "loss": loss,
                "pred_idx": pred_idx.detach(),
                "ntokens": ntokens,
                "nll": dec_outdict["nll"],
                "src_nopad_mask": src_nopad_mask,
                "loss_theta":quantizer_out["loss_theta"],
                "cludict": self.cludict
            }
        )
        return result

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

