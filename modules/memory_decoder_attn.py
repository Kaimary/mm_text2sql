import torch
from allennlp import nn
from allennlp.modules import Attention
from torch.nn import Parameter as Param
from allennlp.nn import util

class MemDecoderAttn(torch.nn.Module):
    def __init__(self,
                 attention: Attention,
                 embedding_dim,
                 nhop):
        #
        # Input
        #  embedding_dim : number of embedding dimentions in memory
        #  nhop: number of memory hops
        #
        super(MemDecoderAttn, self).__init__()

        self.attention = attention
        self.nhop = nhop
        self.hidden = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor):
        #
        # Input
        #  Q : Matrix of queries: (batch_size, num_question_toks, embedding_dim)
        #  K : Matrix of keys: (batch_size, num_memories, embedding_dim)
        #  V : Matrix of values: (batch_size, num_memories, embedding_dim)
        #
        # Output
        #  R : soft-retrieval of values: (batch_size, num_question_toks, embedding_dim)
        #  attn_weights : soft-retrieval of values: (batch_size, num_question_toks, num_memories)
        Q_hop = Q
        for h in range(self.nhop):
            attn_weights = self.attention(Q_hop, K, None)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=2)
            R = util.weighted_sum(V, attn_weights)
            concat_by_step = torch.cat((Q_hop, R), 1)
            # (batch_size, num_question_toks, embedding_dim)
            Q_hop = self.tanh(self.hidden(concat_by_step))

        return Q_hop, attn_weights



