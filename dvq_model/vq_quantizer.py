import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from dvq_model.util import PackedSequneceUtil


class VectorQuantizer(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()

        # K
        self._num_embeddings = num_embeddings #256
        # D
        self._embedding_dim = embedding_dim #64
        self.commitment_cost = config.vq.commitment_cost #0.001
        self.ema = config.vq.use_ema
        # K × D
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # 50 * 64
        if self.ema:
            self.ema_init()
        self.load_topic_embedding = 1
    def forward(self, inputs, topic_embedding, theta, pretrain_vq):
        if pretrain_vq and self.load_topic_embedding:
            self._embedding.weight = topic_embedding
            self.load_topic_embedding = 0
        # support PackedSequence
        packed_seq_util = PackedSequneceUtil()
        inputs = packed_seq_util.preprocess(inputs) 
        input_shape = inputs.shape  
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        min_distances, encoding_indices = torch.min(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self._num_embeddings).float() 
        quantized = self._embedding(encoding_indices).view(input_shape)
        quantized_st = inputs + (quantized - inputs).detach()
        if self.ema and self.training:
            self.ema_update(encodings, flat_input)  
        #encodings
        encodings =  packed_seq_util.postprocess(encodings, pad=0.0) 
        distances = packed_seq_util.postprocess(distances, pad=0.0) 
        encodings =  distances + (encodings-distances).detach()
        encodings = torch.sum(encodings,dim=1) 
        outputs_prob = torch.softmax(encodings, -1)
        
        # calculate loss
        loss_vq = F.mse_loss(quantized, inputs.detach(), reduction="sum")
        loss_commit = F.mse_loss(inputs, quantized.detach(), reduction="sum")
        if self.ema:
            loss = loss_commit * self.commitment_cost
        else:
            loss = loss_commit * self.commitment_cost + loss_vq

        if packed_seq_util.is_packed:
            quantized_st = packed_seq_util.postprocess(quantized_st, pad=0.0)
            encoding_indices = packed_seq_util.postprocess(encoding_indices, pad=-1)
            min_distances = packed_seq_util.postprocess(min_distances, pad=-1)
        else:
            encoding_indices = encoding_indices.contiguous().view(input_shape[:-1])
            min_distances = min_distances.contiguous().view(input_shape[:-1])

        output_dict = {
            "quantized": quantized_st,
            "loss": loss,
            "encoding_indices": encoding_indices,
            "min_distances": min_distances,
            "loss_commit": loss_commit,
            "loss_theta": 0,
        }
        return output_dict

    def ema_init(self):
        self._decay = config.vq.ema.decay
        self._epsilon = config.vq.ema.epsilon
        # K
        self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings))
        # (K, D)
        self.register_buffer(
            "_ema_w", torch.Tensor(self._num_embeddings, self._embedding_dim)
        )
        self._ema_w.data = self._embedding.weight.clone()

    def ema_update(self, encodings, flat_input):
        with torch.no_grad():
            # N moving average
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # additive smoothing to avoid zero count
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            # m moving average
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            # e update
            self._embedding.weight.data.copy_(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )


class DVQ(nn.Module):
    def __init__(
        self, config, num_embeddings, embedding_dim, split=2, decompose_option="slice"
    ):
        super().__init__()

        self.K = num_embeddings 
        self.D = embedding_dim 
        self.M = split 
        self.decompose_option = decompose_option

        if self.decompose_option == "project":
            self.projection = nn.Linear(embedding_dim * split, embedding_dim * split)

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(config, num_embeddings=self.K, embedding_dim=self.D)
                for _ in range(self.M)
            ]
        )

    def decompose(self, inp, option="slice"):
        # each slice: B × T (optional) × D

        # support PackedSequence
        is_packed = isinstance(inp, PackedSequence)
        if is_packed:
            inp, *pack_shape = inp
        if option == "project":
            inp = self.projection(inp)
        elif option == "slice":
            pass

        slices = inp.split(self.D, dim=-1)
        if is_packed:
            slices = [PackedSequence(i, *pack_shape) for i in slices]
        return slices

    def forward(self, inputs,  topic_embedding, theta, pretrain_vq):
        """
        inputs: B × T (optional) × (M * D)
        """
        slices = self.decompose(inputs, self.decompose_option) 
        
        
        # apply vq to each slice separately
        vq_out_list = []
        for slice, vq_layer in zip(slices, self.vq_layers):
            vq_out = vq_layer(slice, topic_embedding, theta, pretrain_vq)  
            vq_out_list.append(vq_out) 
        aggregate_out = {}
        keys = vq_out_list[0].keys()
        for k in keys:
            aggregate_out[k] = []
            for vq_out in vq_out_list:
                aggregate_out[k].append(vq_out[k])
        
        quantized = torch.cat(aggregate_out["quantized"], dim=-1)
        loss = torch.stack(aggregate_out["loss"]).sum()
        loss_commit = torch.stack(aggregate_out["loss_commit"]).sum()
        encoding_indices = torch.stack(aggregate_out["encoding_indices"], dim=-1)

        # combine by stacking, can do sum or mean later on
        quantized_stack = torch.stack(aggregate_out["quantized"], dim=-2) 

        loss_theta = aggregate_out["loss_theta"][0]
        output_dict = {
            "quantized": quantized,
            "quantized_stack": quantized_stack,
            "encoding_indices": encoding_indices,
            "loss": loss,
            "loss_commit": loss_commit.detach(),
            "min_distances": None,
            "loss_theta": loss_theta,
        }

        return output_dict
